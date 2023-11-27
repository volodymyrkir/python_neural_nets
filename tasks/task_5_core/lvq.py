import numpy as np


class LVQ:
    def __init__(self):
        pass

    @staticmethod
    def distance(first, second):
        return np.sqrt(np.sum(np.square(first - second), axis=1))

    def update_two_step(self, train_data, alpha):
        for features, label in zip(train_data[0], train_data[1]):
            distances = self.distance(self.patterns, features)

            min_idx, right_idx = 0, 0
            for i in range(distances.shape[0]):
                if distances[i] < distances[min_idx]:
                    min_idx = i

                if distances[i] < distances[right_idx] and self.labels[i] == label:
                    right_idx = i

            if min_idx == right_idx:
                bmu_w = self.patterns[min_idx]
                error = (features - bmu_w)
                bmu_w += error * alpha
            else:
                bmu_w = self.patterns[min_idx]
                error_bmu = (features - bmu_w)
                bmu_w -= error_bmu * alpha

                right_w = self.patterns[right_idx]
                error = (features - right_w)
                right_w += error * alpha

    def update_classic(self, train_data, alpha):
        for features, label in zip(train_data[0], train_data[1]):
            min_idx = np.argmin(self.distance(features, self.patterns))
            bmu_w, bmu_l = self.patterns[min_idx], self.labels[min_idx]
            error = (features - bmu_w)

            if bmu_l == label:
                bmu_w += error * alpha
            else:
                bmu_w -= error * alpha

    def train(self,init_data, train_data, learning_rule='classic', alpha=0.01, epochs=1):
        self.patterns, self.labels = init_data
        for _ in range(epochs):
            if learning_rule == 'classic':
                self.update_classic(train_data, alpha)
            elif learning_rule == 'two_step':
                self.update_two_step(train_data, alpha)
            else:
                raise ValueError("Not supported train function")

    def predict(self, entries):
        return list(map(lambda entry: self.labels[np.argmin(self.distance(entry, self.patterns))], entries))
