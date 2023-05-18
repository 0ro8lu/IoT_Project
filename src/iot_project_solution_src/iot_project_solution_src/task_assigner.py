 
import time
import random
import math

from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from rosgraph_msgs.msg import Clock
from iot_project_interfaces.srv import TaskAssignment
from iot_project_solution_interfaces.action import PatrollingAction
from iot_project_interfaces.msg import TargetsTimeLeft
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

from sklearn.cluster import KMeans
import numpy as np
import statistics

class TaskAssigner(Node):

    def __init__(self):

        super().__init__('task_assigner')

            
        self.task = None
        self.no_drones = 0
        self.targets = []
        self.thresholds = []


        self.action_servers = []
        self.current_tasks =  []
        self.idle = []

        self.drone_assigned_points = None
        self.current_thresholds = []
        self.drone_positions = None
        self.drone_first_assignment = None
        self.aoi = None
        self.fairness = None
        self.violation = None
        self.drone_outliers = None

        self.visiting_target = None

        self.sim_time = 0

        self.task_announcer = self.create_client(
            TaskAssignment,
            '/task_assigner/get_task'
        )

        self.sim_time_topic = self.create_subscription(
            Clock,
            '/world/iot_project_world/clock',
            self.store_sim_time_callback,
            10
        )



    # Function used to wait for the current task. After receiving the task, it submits
    # to all the drone topics
    def get_task_and_subscribe_to_drones(self):

        self.get_logger().info("Task assigner has started. Waiting for task info")

        while not self.task_announcer.wait_for_service(timeout_sec = 1.0):
            time.sleep(0.5)

        self.get_logger().info("Task assigner is online. Requesting patrolling task")

        assignment_future = self.task_announcer.call_async(TaskAssignment.Request())
        assignment_future.add_done_callback(self.first_assignment_callback)



    # Callback used for when the patrolling task has been assigned for the first time.
    # It configures the task_assigner by saving some useful values from the response
    # (more are available for you to read and configure your algorithm, just check
    # the TaskAssignment.srv interface).
    # The full response is saved in self.task, so you can always use that to check
    # values you may have missed. Or just save them here by editing this function.
    # Once that is done, it creates a client for the action servers of all the drones
    def first_assignment_callback(self, assignment_future):

        task : TaskAssignment.Response = assignment_future.result()

        self.task = task
        self.no_drones = task.no_drones
        self.targets = task.target_positions
        self.thresholds = task.target_thresholds
        self.aoi = task.aoi_weight
        self.fairness = task.fairness_weight
        self.violation = task.violation_weight

        self.current_tasks = [None]*self.no_drones
        self.idle = [True] * self.no_drones

        self.drone_positions = [[] for _ in range(self.no_drones)]
        self.drone_first_assignment = [False for _ in range(self.no_drones)]


        # Now create a client for the action server of each drone
        for d in range(self.no_drones):
            self.action_servers.append(
                ActionClient(
                    self,
                    PatrollingAction,
                    'X3_%d/patrol_targets' % d,
                )
            )

        # Now create a subscriber to the odometry topic for each drone in order to retrieve the current position
        for d in range(self.no_drones):
            self.create_subscription(
            Odometry,
            "/X3_{0}/odometry".format(d),
            lambda msg, d=d: self.odometry_callback(msg, d),
            10)

        self.update_thresholds = self.create_subscription(TargetsTimeLeft, "/task_assigner/targets_time_left", self.thresholds_callback, 10)

        while self.drone_positions[0] == []:
            continue

        # Sort target based on their thresholds
        merged_list = list(zip(self.targets, self.thresholds))
        sorted_list = sorted(merged_list, key=lambda x: x[1])
        sorted_targets, sorted_thresholds = zip(*sorted_list)

        if self.no_drones > len(self.targets):
            self.drone_assigned_points = self.assign_drone_to_target(sorted_targets)
        else:
             self.define_clusters()




    # This method starts on a separate thread an ever-going patrolling task, it does that
    # by checking the idle state value of every drone and submitting a new goal as soon as
    # that value goes back to True
    def keep_patrolling(self):

        #Function used to perform a fair patrolling. Each drone will loop on its cluster.
        def fair_patrolling():
            while True:
                for drone_id in range(self.no_drones):
                    if self.idle[drone_id] and self.drone_assigned_points is not None: # Wait that the cluster has been computed
                        Thread(target=self.submit_task, args=(drone_id, self.drone_assigned_points[drone_id])).start()
                        

                time.sleep(0.1)

        '''
        per ogni drone, calcolare uno score per ogni punto dello spazio:
         - normalizzare distanze-thresholds (-1,1)
           - -1 threshold più basso
           - -1 distanza più bassa
         - sommare i valori normalizzati
         - ordinare le somme in modo crescente
         - verificare che ogni drone vada in un punto diverso
        '''
        
        def unfair_patrolling():

            while self.drone_assigned_points is None:
                continue

            last_visit= [None for _ in range(self.no_drones)]

            while True:
                for drone_id in range(self.no_drones):
                    if self.idle[drone_id]:

                        scores = self.get_scores(drone_id, last_visit[drone_id])

                        if drone_id == 1:
                            print("DRONE {} \n\nSCORES {}".format(drone_id, scores))

                        max_score = max(scores)
                        target_index = scores.index(max_score)
                        target = self.drone_assigned_points[drone_id][target_index]

                        last_visit[drone_id] = target_index

                        Thread(target=self.submit_task, args=(drone_id, [target])).start()
                
                time.sleep(0.1)


        if self.fairness is not None and self.fairness >= 0.5:
            Thread(target=fair_patrolling).start() # Use fair patrolling algorithm
        else:
            Thread(target=unfair_patrolling).start() # Use unfair patrolling algorithm

    

    # Submits a patrol task to a single drone. Basic implementation just takes the array
    # of targets and shuffles it. Is up to you to improve this part and come up with your own
    # algorithm.
    # 
    # TIP: It is highly suggested to start working on a better scheduling of the targets from here.
    #      some drones may want to inspect only a portion of the nodes, other maybe more.
    #
    #      You may also implement a reactive solution which checks for the target violation
    #      continuously and schedules precise tasks at each step. For that, you can call again
    #      the task_announcer service to get an updated view of the targets' state; the last
    #      visit of each target can be read from the array last_visits in the service message.
    #      The simulation time is already stored in self.sim_time for you to use in case
    #      Times are all in nanoseconds.
    def submit_task(self, drone_id, targets_to_patrol = None):

        self.get_logger().info("Submitting task for drone X3_%s" % drone_id)
    
        while not self.action_servers[drone_id].wait_for_server(timeout_sec = 1.0):
            return

        self.idle[drone_id] = False

        #if not targets_to_patrol:
        #    targets_to_patrol = self.targets.copy()
        #    random.shuffle(targets_to_patrol)

        patrol_task =  PatrollingAction.Goal()
        patrol_task.targets = targets_to_patrol

        patrol_future = self.action_servers[drone_id].send_goal_async(patrol_task)

        # This is a new construct for you. Basically, callbacks have no way of receiving arguments except
        # for the future itself. We circumvent such problem by creating an inline lambda functon which stores
        # the additional arguments ad-hoc and then calls the actual callback function
        patrol_future.add_done_callback(lambda future, d = drone_id : self.patrol_submitted_callback(future, d))



    # Callback used to verify if the action has been accepted.
    # If it did, prepares a callback for when the action gets completed
    def patrol_submitted_callback(self, future, drone_id):

        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info("Task has been refused by the action server")
            return
        
        result_future = goal_handle.get_result_async()

        # Lambda function as a callback, check the function before if you don't know what you are looking at
        result_future.add_done_callback(lambda future, d = drone_id : self.patrol_completed_callback(future, d))



    # Callback used to update the idle state of the drone when the action ends
    def patrol_completed_callback(self, future, drone_id):
        self.get_logger().info("Patrolling action for drone X3_%s has been completed. Drone is going idle" % drone_id)
        self.idle[drone_id] = True



    # Callback used to store simulation time
    def store_sim_time_callback(self, msg):
        self.sim_time = msg.clock.sec * 10**9 + msg.clock.nanosec



    # Function used to calculate the target clusters
    def define_clusters(self):

        points = []
        for target in self.targets: # Create a matrix that can be used by K-means
            points.append([target.x, target.y, target.z])

        n_clusters = self.no_drones # The number of clusters will be equal to the number of drones

        kmeans = KMeans(n_clusters=n_clusters) # Creating the KMeans object with specified number of clusters
        kmeans.fit(points) # Ecxecuting kmeans on all points
        tmp_list = [[] for _ in enumerate(kmeans.labels_)] # Making an empty list for points given to every cluster.

        for i, label in enumerate(kmeans.labels_): # Assigning every point to the corresponding cluster
            tmp_list[label].append(Point(x=points[i][0], y=points[i][1], z=points[i][2]))

        centroids = [Point(x=center[0], y=center[1], z=center[2]) for center in kmeans.cluster_centers_] # Obtain the centroid lists
        
        # Obtain the list with (centroid, list of points of the cluster)
        points_by_centroid = [[] for _ in range(n_clusters)]  # Initialize the list
        for i, centroid in enumerate(centroids): # For each cluster and each centroid
            centroid_point = centroid
            cluster_points = tmp_list[i]
            points_by_centroid[i] = (centroid_point, cluster_points) # Assign the centroid and its cluster


        self.drone_assigned_points = [[] for _ in range(self.no_drones)] # Initialize the list

        centroid_drone_list = self.assign_drone_to_target(centroids) # Obtain the list which associates each drone to a centroid
        for drone_id in range(self.no_drones):

            assigned_centroid = centroid_drone_list[drone_id][0] # Obtain the centroid assinged to the drone
            for x in points_by_centroid: # Loop on the list to find the list of points associated to the centroid
                if x[0] == assigned_centroid: # We have found the assigned centroid inside the list
                    assigned_cluster = x[1] # Save the list of points associated to that centroid (cluster)
                    break
            self.drone_assigned_points[drone_id] = assigned_cluster # Save the list of points into the final list
            

        #Sort the points in each cluster based on the initial threshold
        for drone_id in range(self.no_drones):
            self.drone_assigned_points[drone_id].sort(key=lambda x: self.targets.index(x))


    # Function used to assign a drone to the nearest target
    def assign_drone_to_target(self, target_list):

        not_assigned_targets = target_list.copy() # List of the drones that are without an assigned cluster
        target_drone_list = [[] for _ in range(self.no_drones)]

        # Assign the drone to the nearest target
        for drone in range(self.no_drones):
            min_dist = None
            for target in not_assigned_targets:
                current_dist = self.euclidean_distance_3d(target, self.drone_positions[drone][0]) # Function to calculate eucledian distance in 3D space

                if min_dist is None: # If we are at the beginning of the search
                    min_dist = current_dist # Update minimum distance
                    selected_target = target # Update selected target

                elif min_dist > current_dist: # If the current drone is nearest to the centroid
                    min_dist = current_dist # Update the minimum distance
                    selected_target = target # Update the target
            
            target_drone_list[drone] = [selected_target] # The cluster is a single point (the nearest tharget)
            not_assigned_targets.remove(selected_target) # Remove the target from the list since now it has an assigned target

        return target_drone_list


    def get_distances_from_drone(self, drone_id):
        
        distances = []
        for target in self.drone_assigned_points[drone_id]:
            distances.append(self.euclidean_distance_3d(target, self.drone_positions[drone_id][0]))

        return distances

    # Function used to calcuate the eculidian distance between 2 points in a 3D space
    def euclidean_distance_3d(self, point1, point2):
        x1, y1, z1 = point1.x, point1.y, point1.z
        x2, y2, z2 = point2.x, point2.y, point2.z
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


    def normalize(self, values: list):
        min_value = min(values)
        max_value = max(values)
        range_value = max_value - min_value
        if range_value == 0:
            return values

        normalized_values = [(value - min_value) / range_value for value in values]
        return normalized_values
    

    def get_scores(self, drone_id, last_visit):
        
        thresholds = []
        for i in range(len(self.current_thresholds)):
            if self.targets[i] in self.drone_assigned_points[drone_id]:
                thresholds.append(self.current_thresholds[i])
                        
        norm_thresholds = self.normalize(thresholds)
        norm_distances = self.normalize(self.get_distances_from_drone(drone_id))

        scores = [0 for _ in range(len(self.drone_assigned_points[drone_id]))]

        for i in range(len(self.drone_assigned_points[drone_id])):
            if last_visit is not None and last_visit == i:
                scores[i] = -100
            else:
                scores[i] = 1 - ((1 - self.violation / 10) * norm_thresholds[i]) - norm_distances[i]

        return scores

    # Callback function used to obtain the drone real time position, each element in the list is a list [position, orientation]
    def odometry_callback(self, msg:Odometry, drone_id):

        self.drone_positions[drone_id] = [msg.pose.pose.position, msg.pose.pose.orientation]


    # Callback function used to obtain the current threasholds of the targets
    def thresholds_callback(self, msg:TargetsTimeLeft):
        self.current_thresholds = msg.times

def main():

    time.sleep(3.0)
    
    rclpy.init()

    task_assigner = TaskAssigner()
    executor = MultiThreadedExecutor()
    executor.add_node(task_assigner)

    task_assigner.get_task_and_subscribe_to_drones()
    task_assigner.keep_patrolling()

    executor.spin()

    executor.shutdown()
    task_assigner.destroy_node()

    rclpy.shutdown()

