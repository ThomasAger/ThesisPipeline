

####### MANAGING VECTORS #######

import math


def magnitude(v):
    return math.sqrt(sum(v[i] * v[i] for i in range(len(v))))


def add(u, v):
    return [u[i] + v[i] for i in range(len(u))]


def sub(u, v):
    return [u[i] - v[i] for i in range(len(u))]


def dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


def normalize(v):
    vmag = magnitude(v)
    return [v[i] / vmag for i in range(len(v))]


def scaleSpaceUnitVector(space, file_name):
    space = np.asarray(space).transpose()
    print(len(space), len(space[0]))
    scaled_vector = []
    for v in space:
        if np.sum(v) != 0:
            norm = normalize(v)
            scaled_vector.append(norm)
        else:
            scaled_vector.append(v)
    space = space.transpose()
    write2dArray(scaled_vector, file_name)


def scaleSpace(space, lower_bound, upper_bound, file_name):
    minmax_scale = MinMaxScaler(feature_range=(lower_bound, upper_bound), copy=True)
    space = minmax_scale.fit_transform(space)
    write2dArray(space, file_name)
    return space

def plotSpace(space):

    single_values = []

    counter = 0
    for s in space:
        single_values.extend(s)

    # basic plot
    sns.distplot(single_values, kde=False, rug=False)
    sns.plt.show()
    print ("now we here")