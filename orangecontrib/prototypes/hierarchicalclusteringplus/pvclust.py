from math import sqrt
import numpy as np
from multiprocessing.pool import ThreadPool as Pool

from scipy.stats import norm
from scipy.cluster.hierarchy import to_tree
from sklearn.linear_model import LinearRegression

from orangecontrib.prototypes.hierarchicalclusteringplus.hierarchical_plus\
    import dist_matrix_linkage


class PvClust:
    """
    The algorithm is firstly implemented in R by Suzuki and Shimodira (2006):
    Pvclust: an R package for assessing the uncertanity in hierarchical
    clustering. This is Python reimplementation. The final values are
    Approximately Unbiased p-value (AU) and Bootstrap Probability (BP) which
    are reporting the significance of each cluster in dendrogram. The AU value
    is less biased and clusters that have this value greater than 95 are
    considered significant.

    Both values are calculated using Multiscale Bootstrap Resampling.
    """
    def __init__(self, data, method, metric, nboot,
                 r=np.array(range(5, 15)), normalize=False, callback=None):
        """
        :param data: a dataset on which clustering and sampling is done
        :param method: a linkage method used in hierarchical clustering
        :param metric: a distance metric used in hierarchical clustering
        :param nboot: a number of bootstrap samples
        :param r: an array of scaling constants
        :param normalize: boolean value reporting if data should be normalized
         when calculating distances

        :returns Approximately Unbiased p-value and Bootstrap Probability for
        each dendrogram node.
        :rtype dict
        """
        self.data = data
        self.nboot = nboot  # number of bootstrap replicates
        self.metric, self.method = metric, method
        self._normalize = normalize
        self._callback = callback

        n = len(self.data)
        r = np.array([i/10 for i in r])
        self.n_scaled = np.unique([int(i*n) for i in r])
        self.r = np.unique([i/n for i in self.n_scaled])

        # apply hierarchical clustering and get clusters
        hc = HierarchicalClusteringClusters(
            data.transpose(data), self.method, self.metric, self._normalize)
        self.clusters = hc.find_clusters()
        self.result = self._wls_fit()

    def _hc(self, n):
        """ Do bootstrap sampling and then apply hierarchical clustering to
        the sample """
        data = self.data
        # we are sampling instances
        data = data[np.random.randint(len(data), size=n), :]
        # HC is applied to columns each time (hence transposing)
        temp = HierarchicalClusteringClusters(
            data.transpose(data), self.method, self.metric, self._normalize)
        clusters = temp.find_clusters()

        return clusters

    def _n_bootstrap_probability(self, i, n):
        """ Calculate bootstrap probability of each cluster for the dataset of
        size n """
        # counting repetitions of the same clusters throughout nboot different
        # clusterings
        self._call_callback(i*10)
        repetitions = {c: 0 for c in range(len(self.clusters))}

        # apply HC nboot times on dataset of size n
        for _ in range(self.nboot):
            sample_clusters = self._hc(n)
            # compare obtained clusters with the main ones and update
            # repetitions if necessary
            for cluster in sample_clusters:
                if cluster in self.clusters:
                    repetitions[self.clusters.index(cluster)] += 1

        # calculate BP probability for each cluster for the sample of size n
        BP = [repetitions[k]/self.nboot for k in repetitions.keys()]

        return np.array(BP)

    def _call_callback(self, i):
        return self._callback(i)

    def _table(self):
        """ Make a table of bootstrap probabilities for each sample size"""
        with Pool() as pool:
            probabilities = pool.starmap(self._n_bootstrap_probability,
                                         enumerate(self.n_scaled))
        table = probabilities[0]
        for i in probabilities[1::]:
            table = np.column_stack((table, i))

        return table

    # Use this function for non-parallel version
    # def _table(self):
    #     """ Make a table of bootstrap probabilities for each sample size"""
    #     # table = np.empty([len(self.data.transpose(self.data))-1, 1])
    #     table = np.empty([len(self.clusters), 1])
    #
    #     # for each sample size calculate BPs of all clusters and
    #     # add it to the table
    #     for i in range(len(self.n_scaled)):
    #         temp = self._n_bootstrap_probability(self.n_scaled[i])
    #         table = np.column_stack((table, temp))
    #         self._call_callback(i*10)
    #     table = np.delete(table, 0, 1)
    #
    #     return table

    def _wls_fit(self):
        """ Take all calculated bootstrap probabilities of a single cluster and
        fit a curve to it in order to calculate AU and BP for that cluster"""
        nboot, r = self.nboot, self.r
        r_sq_org = np.array([sqrt(j) for j in r])
        r_isq_org = np.array([1/sqrt(j) for j in r])
        eps = 0.001
        table = self._table()

        result = {i:(0,0) for i in range(int(len(table)/2)+1)}
        for i in range(int(len(table)/2)+1, len(table)):
            BP = table[i]
            nboot_list = np.repeat(nboot, len(BP))
            use = np.logical_and(np.greater(BP, eps), np.less(BP, 1-eps))
            if sum(use) < 3:
                result[i] = (0, 0) if np.mean(BP) < 0.5 else (1, 1)
            else:
                BP = BP[use]
                r_sq = r_sq_org[use]
                r_isq = r_isq_org[use]
                nboot_list = nboot_list[use]

                y = -norm.ppf(BP) # change of BP values
                X_model = np.array([[i, j] for i, j in zip(r_sq, r_isq)])
                weight = ((1-BP)*BP)/((norm.pdf(y)**2)*nboot_list)
                # fitting the curve using WLS to the observed change
                model = LinearRegression(fit_intercept=False)
                results_lr = model.fit(X_model, y, sample_weight=1/weight)
                z_au = np.array([1, -1])@results_lr.coef_
                z_bp = np.array([1, 1])@results_lr.coef_
                # AU and BP
                result[i] = (1-norm.cdf(z_au), 1-norm.cdf(z_bp))
                result[i] = tuple(round(i, 2) for i in result[i])

        return result

class HierarchicalClusteringClusters:
    """Apply Hierarchical Clustering on the data and find elements of
    each cluster.
    """
    def __init__(self, data, method, metric, normalize=False):
        """
        :param data: a dataset on which clustering and sampling is done
        :param method: a linkage method used in hierarchical clustering
        :param metric: a distance metric used in hierarchical clustering
        :param normalize: boolean value reporting if data should be normalized
        when calculating distances
        """
        # always cluster columns (axis=1)
        dist_matrix = metric(data, axis=1, impute=True, normalize=normalize)
        linkage_matrix = dist_matrix_linkage(dist_matrix, method)
        self.nodes = to_tree(linkage_matrix, rd=True)[1]

    def l_branch(self, left, node, nodes):
        if not node.is_leaf():
            if node.left.id > (len(nodes)-1)/2:
                self.l_branch(left, nodes[node.left.id], nodes)
                self.r_branch(left, nodes[node.left.id], nodes)
            else:
                left.append(node.left.id)
        else:
            left.append(node.id)

        return list(set(left))

    def r_branch(self, right, node, nodes):
        if not node.is_leaf():
            if node.right.id > (len(nodes)-1)/2:
                self.r_branch(right, nodes[node.right.id], nodes)
                self.l_branch(right, nodes[node.right.id], nodes)
            else:
                right.append(node.right.id)
        else:
            right.append(node.id)

        return list(set(right))

    # find all clusters produced by HC from leaves to the root node
    def find_clusters(self):
        nodes = self.nodes
        clusters = []
        for i in range(len(nodes)):
        # for i in range(len(sch.leaves_list(self.linkage_matrix)), len(nodes)):
            left = self.l_branch([], nodes[i], nodes)
            right = self.r_branch([], nodes[i], nodes)

            node_i = sorted(set(left + right))
            if node_i:
                clusters.append(node_i)

        return clusters
