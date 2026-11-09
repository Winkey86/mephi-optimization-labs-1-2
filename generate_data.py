"""Regenerate synthetic teaching data with NumPy PCG64 seed 10119."""
from common import ROOT, SEED, np, save_json


def generate():
    rng = np.random.default_rng(SEED)
    A = rng.integers(1, 7, (3, 8)) / 10
    demand = rng.integers(12, 25, 8)
    save_json(ROOT / 'lab1/data/variant.json', {
        'N': 19, 'Q': 1, 'Y': 26, 'V': 17, 'seed': SEED,
        'products': ['Батон', 'Хлеб ржаной', 'Хлеб зерновой', 'Булочка',
                     'Круассан', 'Слойка', 'Пирожок', 'Сдоба'],
        'oven_hours_per_batch': A, 'oven_hours': np.round(A @ (demand * .45), 2),
        'margin_rub_per_batch': rng.integers(300, 701, 8),
        'max_batches': demand, 'min_batches': [5, 4, 3, 0, 0, 0, 0, 0]})
    rng = np.random.default_rng(SEED)
    demand = rng.integers(5, 10, 5) * 5
    supply = np.ceil(demand.sum() * np.array([.32, .35, .30, .38])).astype(int)
    cost = np.empty((4, 5, 2), int)
    cost[:, :, 0] = rng.integers(800, 1601, (4, 5))
    cost[:, :, 1] = rng.integers(350, 801, (4, 5))
    capacity = np.empty((4, 5, 2), int)
    capacity[:, :, 0] = rng.integers(18, 31, (4, 5))
    capacity[:, :, 1] = rng.integers(6, 16, (4, 5))
    save_json(ROOT / 'lab2/data/variant.json', {
        'N': 19, 'Q': 1, 'Y': 26, 'V': 17, 'seed': SEED,
        'warehouses': ['С1', 'С2', 'С3', 'С4'], 'customers': ['М1', 'М2', 'М3', 'М4', 'М5'],
        'modes': ['Авто', 'ЖД'], 'supply_t': supply, 'demand_t': demand,
        'cost_rub_per_t': cost, 'capacity_t': capacity})


if __name__ == '__main__':
    generate()
