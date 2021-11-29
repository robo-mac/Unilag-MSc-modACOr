def get_probability(self, d, y, x, n, c):
    """
    This gets the probability of drop / pickup for any given Datum, d
    :param d: the datum
    :param x: the x location of the datum / ant carrying datum
    :param y: the y location of the datum / ant carrying datum
    :param n: the size of the neighbourhood function
    :param c: constant for convergence control
    :return: the probability of
    """
    # Starting x and y locations
    y_s = y - n
    x_s = x - n
    total = 0.0
    # For each neighbour
    for i in range((n*2)+1):
        xi = (x_s + i) % self.dim[0]
        for j in range((n*2)+1):
        # If we are looking at a neighbour
            if j != x and i != y:
                yj = (y_s + j) % self.dim[1]
                # Get the neighbour, o
                o = self.grid[xi][yj]
                # Get the similarity of o to x
            if o is not None:
                s = d.similarity(o)
                total += s
    # Normalize the density by the max seen distance to date
    # (math.pow((n*2)+1, 2) - 1) is the number of points considered
    md = total / (math.pow((n*2)+1, 2) - 1)
    # Update the maximum distance seen
    if md > self.max_d:
        self.max_d = md
    # Compute the density function and return it
    density = total / (self.max_d * (math.pow((n*2)+1, 2) - 1))
    density = max(min(density, 1), 0)
    t = math.exp(-c * density)
    probability = (1-t)/(1+t)
    return probability