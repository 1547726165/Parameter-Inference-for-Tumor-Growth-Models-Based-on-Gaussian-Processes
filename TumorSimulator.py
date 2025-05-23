import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat

class TumorSimulation:
    def __init__(self, nx=30, ny=30, nz=30, dx=1.0, dy=1.0, dz=1.0, 
                 r=1.0, D_w=0.1, D_g=None, num_steps=1000,flag=True):
        # 初始化几何参数
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.r = r
        self.D_w = D_w
        self.D_g = 10*D_w if D_g is None else D_g
        self.num_steps = num_steps
        self.flag=flag
        
        # 计算几何特征
        self.center = np.array([nx//2, ny//2, nz//2])
        self.R1 = min(nx, ny, nz) // 2
        self.R2 = self.R1 // 2
            
        # 初始化场变量
        self.D = np.zeros((nx, ny, nz), dtype=np.double)
        self.D_binary = np.zeros_like(self.D)
        self.u = np.zeros((nx, ny, nz))
            
        # 预计算边界掩膜
        self.boundary_mask = None
        if(self.flag==True):
         # 初始化模拟参数
            self._setup_diffusion_coefficients()
        self._set_initial_conditions()
        self._compute_time_step()
            
        # 数据存储
        self.x_inter_list = []
        self.y_inter_list = []

    def _setup_diffusion_coefficients(self):
        """使用向量化操作设置扩散系数"""
        x, y, z = np.ogrid[:self.nx, :self.ny, :self.nz]
        distance = np.sqrt((x - self.center[0])**2 + 
                         (y - self.center[1])**2 + 
                         (z - self.center[2])**2)
        
        inner_mask = distance <= self.R2
        outer_mask = (distance > self.R2) & (distance <= self.R1)
        
        self.D[inner_mask] = self.D_g
        self.D_binary[inner_mask] = 1
        self.D[outer_mask] = self.D_w
        
        # 预计算边界条件掩膜
        self.boundary_mask = distance > self.R1
        np.save('D.npy', self.D_binary)

    def _set_initial_conditions(self):
        """设置初始肿瘤密度"""
        if(self.flag==True):
            z_point = min(self.nz//2 + self.R2, self.nz-1)
            self.u[self.nx//2, self.ny//2, z_point] = 0.5
        else:
            self.u[50, 100, self.nz//2] = 1.0

    def _compute_time_step(self):
        """计算稳定时间步长"""
        D_max = max(self.D_g, self.D_w)
        self.dt = min(0.01, 0.5*self.dx**2/(6*D_max))
        print(f"Using time step: {self.dt:.4f}")

    def _apply_boundary_conditions(self, u_new):
        """应用Dirichlet边界条件"""
        u_new[self.boundary_mask] = 0
        return u_new

    def run_simulation(self, visualize_interval=100):
        """执行主模拟循环"""
        for step in range(self.num_steps):
            # 计算拉普拉斯项
            laplacian = (np.roll(self.u, -1, 0) - 2*self.u + np.roll(self.u, 1, 0)) / self.dx**2 + \
                        (np.roll(self.u, -1, 1) - 2*self.u + np.roll(self.u, 1, 1)) / self.dy**2 + \
                        (np.roll(self.u, -1, 2) - 2*self.u + np.roll(self.u, 1, 2)) / self.dz**2
            
            # 更新肿瘤密度
            u_new = self.u + self.dt * (self.D * laplacian + self.r * self.u * (1 - self.u))
            if(self.flag==True):
                # 应用边界条件
                u_new = self._apply_boundary_conditions(u_new)
            self.u = np.clip(u_new, 0, 1)
            
            # 数据记录和可视化
            if step % visualize_interval == 0:
                self._record_data(step)
                self._visualize(step)

    def _record_data(self, step):
        """记录当前时间步的数据"""
        current_time = step * self.dt
        x, y, z = np.where(self.u >= 0)
        values = self.u[x, y, z]
        entries = np.column_stack((np.full_like(x, current_time), x, y, z))
        self.x_inter_list.append(entries)
        self.y_inter_list.append(values)

    def _visualize(self, step):
        """生成3D可视化"""
        if(self.flag==True):
            import matplotlib.colors as mcolors

            threshold = 0.001
            mask = self.u >= threshold

            # 创建红色渐变 colormap
            red_cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", ["mistyrose", "red", "darkred"])
            norm_values = (self.u - self.u.min()) / (self.u.max() - self.u.min())
            colors = red_cmap(norm_values)

            alpha = np.where(mask, 0.6, 0.0)
            facecolors = np.concatenate((colors[..., :3], alpha[..., None]), axis=-1)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.voxels(mask, facecolors=facecolors, edgecolor='k', linewidth=0.1)

            ax.set_title(f"Tumor Growth Simulation (Step {step})")
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')

            plt.tight_layout()
            plt.show()
        else:
            # 可视化结果 
            plt.imshow(self.u[:, :, self.nz//2], cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Tumor Density at z = nz/2")
            plt.show()

    def generate_observations(self, num_obs=100, num_pred=100, sigma_e=1e-3):
        """生成观测数据集"""
        # 合并所有记录数据
        x_inter = np.vstack(self.x_inter_list)
        y_inter = np.hstack(self.y_inter_list)
        
        # 生成观测数据
        weights = y_inter / y_inter.sum()
        obs_idx = np.random.choice(len(y_inter), num_obs, p=weights)
        x_obs = x_inter[obs_idx]
        y_obs = y_inter[obs_idx] 
        
        # 生成预测点
        remaining_idx = np.setdiff1d(np.arange(len(y_inter)), obs_idx)
        pred_idx = np.random.choice(remaining_idx, num_pred, replace=False)
        x_pred = x_inter[pred_idx]
        y_pred = y_inter[pred_idx]
        if(self.flag==True):
            # 保存数据集
            savemat('tumor_data.mat', {
                'x_obs': x_obs, 'y_obs': y_obs,
                'x_pred': x_pred, 'y_pred': y_pred,
                'x_inter': x_inter, 'y_inter': y_inter,
                'theta_true': [self.D_w, self.r],
                'sigma_e': sigma_e
            })
        else:
            savemat('F:/tumor_data.mat', {
                'x_obs': x_obs, 'y_obs': y_obs,
                'x_pred': x_pred, 'y_pred': y_pred,
                'x_inter': x_inter, 'y_inter': y_inter,
                'theta_true': [self.D_w, self.r],
                'sigma_e': sigma_e
            })
        print(f"Generated {num_obs} observations and {num_pred} prediction points")
        print(f"Observation range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
'''
if __name__ == "__main__":
    # 示例用法
    simulator = TumorSimulation(
        nx=40, ny=40, nz=40,
        D_w=0.2, 
        r=0.8,
        num_steps=500
    )
    
    simulator.run_simulation(visualize_interval=50)
    simulator.generate_observations(num_obs=200, num_pred=300)
'''