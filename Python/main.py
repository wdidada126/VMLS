import numpy as np
from vmls import *

def main():
    print("=== 线性代数及其应用参考代码演示 ===\n")
    
    print("1. 线性组合示例 (Chapter 01)")
    coef = [2, 3]
    vectors = [np.array([1, 2]), np.array([3, 4])]
    result = linear_combination(coef, vectors)
    print(f"   2*[1,2] + 3*[3,4] = {result}")
    
    result_compact = compact_linear_combination(coef, vectors)
    print(f"   使用紧凑方法: {result_compact}\n")
    
    print("2. 标准化示例 (Chapter 03)")
    x = np.array([1, 2, 3, 4, 5])
    x_std = standardize(x)
    print(f"   原始数据: {x}")
    print(f"   标准化后: {x_std}")
    print(f"   均值: {np.mean(x_std):.10f}, 标准差: {np.std(x_std):.10f}\n")
    
    print("3. 相关系数示例 (Chapter 03)")
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 5, 4, 5])
    corr = corr_coef(a, b)
    print(f"   a 和 b 的相关系数: {corr:.4f}\n")
    
    print("4. K-means 聚类示例 (Chapter 04)")
    np.random.seed(42)
    X = np.random.randn(50, 2)
    X[:25] += 2
    assignment, reps, progress = kmeans1(X, num_clusters=2, max_iters=10, random_seed=42)
    print(f"   聚类完成，最终目标函数值: {progress[-1][0]:.4f}\n")
    
    print("5. Gram-Schmidt 正交化示例 (Chapter 05)")
    a = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])]
    q = gram_schmidt(a)
    print(f"   正交化完成，得到 {len(q)} 个正交向量\n")
    
    print("6. QR 分解示例 (Chapter 11)")
    A = np.array([[1, 2], [3, 4], [5, 6]])
    Q, R = QR_factorization(A)
    print(f"   原矩阵 A:\n{A}")
    print(f"   Q 矩阵:\n{Q}")
    print(f"   R 矩阵:\n{R}\n")
    
    print("7. 矩阵的零空间示例 (Chapter 07)")
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ns = nullspace(A)
    print(f"   矩阵 A:\n{A}")
    print(f"   零空间维度: {ns.shape[1]}")
    print(f"   验证 A @ ns ≈ 0: {np.allclose(A @ ns, 0)}\n")

if __name__ == "__main__":
    main()
