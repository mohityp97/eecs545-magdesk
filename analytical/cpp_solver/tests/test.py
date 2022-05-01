# %%
import cppsolver as m
import numpy as np
# %%

assert m.add(1, 2) == 3
assert m.subtract(1, 2) == -1
data = np.load(
    "/home/programming/FinalProject/magsense/Codes/magsense/result/test.npz")
readings = data['data']
gt = data['gt']
# %%
pSensor = 1e-2 * np.array([
    [4.9, -4.9, -1.63],
    [-4.9, -4.9, -1.63],
    [4.9, 4.9, -1.63],
    [-4.9, 4.9, -1.63],
    [4.9, -4.9, 1.63],
    [-4.9, -4.9, 1.63],
    [4.9, 4.9, 1.63],
    [-4.9, 4.9, 1.63],
]).reshape(-1)

# %%
route = []
init_params = np.array([0, 0, 0.1, np.log(2.7), 0, 0, 0.02, 0, 0])

test_readings = []
param = init_params
for i in range(readings.shape[0]):
    reading = readings[i]
    result = m.solve_1mag_jac(reading, pSensor, param)
    param = result
    tmp1 = m.calB(pSensor, param)
    tmp2 = m.calB(pSensor, np.concatenate(
        [result, np.array([1e-2 * -4.9, 1e-2 * 4.9, 1e-2 * 1.63, 0, 0, 1 * np.pi])]))
    test_readings.append(m.calB(pSensor, param))
    # test_readings.append(np.stack(tmp, axis=0))
    route.append(result[4:7])

# %%
route = np.stack(route, axis=0)
test_readings = np.stack(test_readings, axis=0)
# %%
mse = np.mean(np.linalg.norm(route - gt, axis=1, ord=2))
mse2 = np.mean(np.linalg.norm(test_readings - readings, axis=1, ord=2))
# %%
mse
# %%
mse2
