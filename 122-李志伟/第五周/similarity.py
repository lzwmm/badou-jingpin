import random
import sys
import numpy as np

# 初始化聚类
class KMeansCluster:
    # 初始化聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        # 初始化簇
        self.cluster_num = cluster_num
        # 初始化中心点
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]

        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        if (self.points == new_center).all():
            sum = self.__sum_distance(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    # 计算每一列的平均值
    def __center(self, list):
        return np.array(list).mean(axis=0)

    # 欧几里德距离公式
    def __distance(self, p1, p2):
        tem = 0

        for i in range(len(p1)):
            tem += pow(p1[i] - p2[i], 2)
        return pow(tem, 0.5)

    # 计算总距离之和
    def __sum_distance(self, result):
        sum_val = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum_val += self.__distance(result[i][j], self.points[i])
        return sum_val

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数有误")
        # 取下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())

        return np.array(points)

    # 平均距离计算
    def __avg_distance_of_cluster(self, cluster, center):
        if len(cluster) == 0:
            return 0
        try:
            distance_sum = sum([self.__distance(point, center) for point in cluster])
        except Exception as e:
            print(str(e))
        return distance_sum / len(cluster)

    # 类内相似度计算
    def compute_intra_cluster_similarity(self, clusters):
        similarities = []
        for i in range(len(clusters)):
            avg_distance = self.__avg_distance_of_cluster(clusters[i], self.points[i])
            similarity = 1 / (1 + avg_distance)  # 倒数形式，使得小的平均距离对应更高的相似度
            similarities.append(similarity)
        return similarities


x = np.random.rand(100, 9)

Kmeans = KMeansCluster(x, 10)

result, point, sum_val = Kmeans.cluster()
print("result:", result)
print("point:", point)
print("sum_val:", sum_val)
result = [np.random.rand(10, 9).tolist()] * 10
similarities = Kmeans.compute_intra_cluster_similarity(result)
print("Cluster Similarities:", similarities)
