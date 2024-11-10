import torch

class CCA:
    def __init__(self, n_components):
        """
        初始化 CCA 模型
        参数:
        - n_components: 需要计算的典型相关成分数
        """
        self.n_components = n_components

    def fit(self, X, Y):
        """
        计算 X 和 Y 的 CCA 成分和相关系数
        参数:
        - X: 张量, 第一个数据集 (样本数, 特征数1)
        - Y: 张量, 第二个数据集 (样本数, 特征数2)
        
        返回:
        - 相关系数和典型变量
        """
        # 中心化数据
        X -= X.mean(dim=0)
        Y -= Y.mean(dim=0)

        # 计算协方差矩阵
        n = X.size(0)
        Sigma_XX = X.T @ X / (n - 1)
        Sigma_YY = Y.T @ Y / (n - 1)
        Sigma_XY = X.T @ Y / (n - 1)

        # SVD 求解
        Sigma_XX_inv_sqrt = self._matrix_inverse_sqrt(Sigma_XX)
        Sigma_YY_inv_sqrt = self._matrix_inverse_sqrt(Sigma_YY)
        
        # 计算相关矩阵
        T = Sigma_XX_inv_sqrt @ Sigma_XY @ Sigma_YY_inv_sqrt
        U, S, V = torch.svd(T)

        # 取前 n_components 个成分
        self.U = U[:, :self.n_components]
        self.V = V[:, :self.n_components]
        self.correlation = S[:self.n_components]

        return self.correlation, self.U, self.V

    def transform(self, X, Y):
        """
        将 X 和 Y 转换到 CCA 空间
        参数:
        - X: 第一个数据集
        - Y: 第二个数据集
        
        返回:
        - X 和 Y 在 CCA 空间中的表示
        """
        X_c = X @ self.U
        Y_c = Y @ self.V
        return X_c, Y_c

    def _matrix_inverse_sqrt(self, matrix):
        """
        计算矩阵的逆平方根
        参数:
        - matrix: 正定对称矩阵
        
        返回:
        - 矩阵的逆平方根
        """
        eigvals, eigvecs = torch.eig(matrix, eigenvectors=True)
        eigvals_sqrt_inv = torch.diag(1.0 / torch.sqrt(eigvals[:, 0]))
        return eigvecs @ eigvals_sqrt_inv @ eigvecs.T

class SVCCA:
    def __init__(self, n_components):
        """
        初始化 SVCCA
        参数:
        - n_components: 需要保留的奇异向量的数量
        """
        self.n_components = n_components

    def fit(self, X, Y):
        """
        计算 X 和 Y 的 SVCCA 相似性
        参数:
        - X: 第一个数据集 (样本数, 特征数)
        - Y: 第二个数据集 (样本数, 特征数)
        
        返回:
        - 相关系数
        """
        # 中心化数据
        X -= X.mean(dim=0)
        Y -= Y.mean(dim=0)

        # 对 X 和 Y 进行 SVD，并保留前 n_components 个奇异向量
        Ux, Sx, Vx = torch.svd(X)
        X_reduced = Ux[:, :self.n_components] * Sx[:self.n_components]

        Uy, Sy, Vy = torch.svd(Y)
        Y_reduced = Uy[:, :self.n_components] * Sy[:self.n_components]

        # 使用 CCA 计算降维后表示的相似性
        cca = CCA(n_components=self.n_components)
        correlation, _, _ = cca.fit(X_reduced, Y_reduced)

        return correlation
    

class PWCCA:
    def __init__(self, n_components):
        """
        初始化 PWCCA
        参数:
        - n_components: 需要计算的典型相关成分数
        """
        self.n_components = n_components

    def fit(self, X, Y):
        """
        计算 X 和 Y 的 PWCCA 相似性
        参数:
        - X: 第一个数据集 (样本数, 特征数)
        - Y: 第二个数据集 (样本数, 特征数)
        
        返回:
        - 加权相关系数
        """
        # 中心化数据
        X -= X.mean(dim=0)
        Y -= Y.mean(dim=0)

        # 使用 CCA 计算两个数据集之间的相似性
        cca = CCA(n_components=self.n_components)
        correlation, U, V = cca.fit(X, Y)

        # 计算投影权重
        X_projected = X @ U  # 在第一个典型空间中投影
        weights = torch.norm(X_projected, dim=0)
        weights /= weights.sum()  # 归一化权重

        # 加权求和相关系数
        pwcca_score = torch.sum(weights * correlation)

        return pwcca_score