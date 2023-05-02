def check_return_iterable(obj):
    if isinstance(obj, dict):
        return [obj]
    try:
        iter(obj)
        return obj
    except:
        return [obj]


def scale_charges(charges, n_particles):
    net_charge = sum(charges)
    abs_charge = sum([abs(charge) for charge in charges])
    adjust = abs(net_charge) / n_particles
    charges -= (abs(charges) * net_charge/abs_charge)
    return charges
