import torch
import numpy as np
import matplotlib.pyplot as plt

class BATorch_Optimizer(object):
    def __init__(self, device):
        self.device = device

    def optimize_pose(
            self,
            points_xyz, points_uv,
            Twc_estimate, K,
            max_iter
    ):
        xyz = torch.from_numpy(points_xyz)
        xyz = torch.concat((xyz, torch.ones((xyz.shape[0], 1))), dim=1)
        uv = torch.from_numpy(points_uv)
        K = torch.from_numpy(K)

        Twc_tensor = torch.from_numpy(Twc_estimate)
        Twc_tensor.requires_grad = True

        opt = torch.optim.Adam(params=[Twc_tensor], lr=0.1)

        mini_loss, best_pred = np.inf, None
        for idx in range(max_iter):
            opt.zero_grad()

            uv_pred = torch.matmul(torch.matmul(K, Twc_tensor), xyz.T)
            uv_pred = uv_pred.T
            s = uv_pred[:, 2]
            uv_pred[:, 0] = uv_pred[:, 0]/s
            uv_pred[:, 1] = uv_pred[:, 1]/s

            loss = torch.mean(torch.abs(uv_pred[:, :-1] - uv))
            loss.backward()
            opt.step()

            loss_float = loss.item()
            if loss_float<mini_loss:
                mini_loss = loss_float
                best_pred = Twc_tensor

        print("[DEBUG]: loss:%f"%(mini_loss))

        best_pred_np = best_pred.detach().cpu().numpy()
        return best_pred_np

    def bundle_adjust(
            self,
            cameras_points_idx, cameras_uv,
            points_xyz,
            cameras_Twc_estimate,
            cameras_K, max_iter
    ):
        camera_num = cameras_K.shape[0]

        Twcs_tensor = torch.from_numpy(cameras_Twc_estimate)
        Twcs_tensor.requires_grad = True

        points_tensor = torch.from_numpy(points_xyz)
        points_tensor.requires_grad = True

        view_points_idx = []
        cameras_uv_tensor = []
        for camera_id in range(camera_num):
            view_point_id = torch.from_numpy(cameras_points_idx[camera_id]).long()
            view_points_idx.append(view_point_id)

            uv_tensor = torch.from_numpy(cameras_uv[camera_id])
            cameras_uv_tensor.append(uv_tensor)

        cameras_K_tensor = torch.from_numpy(cameras_K)

        opt = torch.optim.Adam(params=[Twcs_tensor, points_tensor], lr=0.1)

        mini_loss, best_Twc, best_Points = np.inf, None, None
        for idx in range(max_iter):
            opt.zero_grad()

            loss = 0.0
            for camera_id in range(camera_num):
                view_points = points_tensor[view_points_idx[camera_id], :]

                K = cameras_K_tensor[camera_id, ...]
                Twc = Twcs_tensor[camera_id, ...]
                uv = cameras_uv_tensor[camera_id]

                uv_pred = torch.matmul(torch.matmul(K, Twc), view_points.T)
                uv_pred = uv_pred.T
                s = uv_pred[:, 2]
                uv_pred[:, 0] = uv_pred[:, 0]/s
                uv_pred[:, 1] = uv_pred[:, 1]/s

                camera_loss = torch.mean(torch.abs(uv_pred[:, :-1] - uv))
                loss += camera_loss

            loss.backward()
            opt.step()

            loss_float = loss.item()
            if loss_float<mini_loss:
                mini_loss = loss_float
                best_Twc = Twcs_tensor
                best_Points = points_xyz

        print("[DEBUG]: loss:%f"%(mini_loss))

        best_Twc = best_Twc.detach().cpu().numpy()
        best_Points = best_Points.detach().cpu().numpy()
        return best_Twc, best_Points

    def poseGraph_opt(self,):
        raise NotImplementedError

    def test_torch_opt(self):
        params = np.random.uniform(1.0, 3.0, (2, ))
        x = np.random.uniform(0.0, 5.0, (300,))
        y = x ** params[0] + params[1] + np.random.random(x.shape[0]) * 3.0

        # plt.scatter(x, y, s=0.5)
        # plt.show()

        pred_tensor = torch.from_numpy(np.random.uniform(1.0, 3.0, (2, )))
        pred_tensor.requires_grad = True

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)

        mini_loss, best_pred = np.inf, 0.
        opt = torch.optim.Adam(params=[pred_tensor], lr=0.1)
        for idx in range(100):
            opt.zero_grad()

            y_pred = x_tensor ** pred_tensor[0] + pred_tensor[1]
            loss = torch.mean(torch.abs(y_pred - y_tensor))
            loss.backward()
            opt.step()

            loss_float = loss.item()
            if loss_float<mini_loss:
                mini_loss = loss_float
                best_pred = pred_tensor

            pred_param_np = pred_tensor.detach().cpu().numpy()
            print("[DEBUG]: loss:%f Estimate: x ** %f + %f "%(loss.item(), pred_param_np[0], pred_param_np[1]))

        best_pred_np = best_pred.detach().cpu().numpy()
        print("[DEBUG]: Original Function: x ** %f + %f "%(params[0], params[1]))
        print("[DEBUG]: loss:%f Pred Function: x ** %f + %f "%(mini_loss, best_pred_np[0], best_pred_np[1]))

if __name__ == '__main__':
    optimizer = BATorch_Optimizer(device='cpu')
    optimizer.test_torch_opt()