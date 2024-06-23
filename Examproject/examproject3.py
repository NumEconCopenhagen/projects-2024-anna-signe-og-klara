class BarycentricInterpolation:

    def r1(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
        upper = (B[1] - C[1]) * (y1 - C[0]) + (C[0] - B[0]) * (y2 - C[1])
        lower = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = upper / lower
        return r   
    
    def r2(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
        upper = (C[1] - A[1]) * (y1 - C[0]) + (A[0] - C[0]) * (y2 - C[1])
        lower = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = upper / lower
        return r
    
    def r3(self, r1, r2):
        r = 1 - r1 - r2
        return r
    
    def distance(self, X, y):
        return np.sqrt((X[:, 0] - y[0])**2 + (X[:, 1] - y[1])**2)
    
    def A_distance(self, X, y):
        condition = X[(X[:, 0] > y[0]) & (X[:, 1] > y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def B_distance(self, X, y):
        condition = X[(X[:, 0] > y[0]) & (X[:, 1] < y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def C_distance(self, X, y):
        condition = X[(X[:, 0] < y[0]) & (X[:, 1] < y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def D_distance(self, X, y):
        condition = X[(X[:, 0] < y[0]) & (X[:, 1] > y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def isin_triangle(self, args, A, B, C):
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        inside = 0 <= r1val <= 1 and 0 <= r2val <= 1 and 0 <= r3val <= 1
        return inside
    
    def y_function(self, args, A, B, C):    
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        y = r1val * A + r2val * B + r3val * C
        return y
    
    def y_function(args, A, B, C):    
        r1val = r1(args, A, B, C)
        r2val = r2(args, A, B, C)
        r3val = r3(r1val, r2val)
        return r1val * A + r2val * B + r3val * C
    
    def approximate_f(y, X, F):
        A = A_distance(X, y)
        B = B_distance(X, y)
        C = C_distance(X, y)
        D = D_distance(X, y)
        if isin_triangle(y, A, B, C):
            r1val = r1(y, A, B, C)
            r2val = r2(y, A, B, C)
            r3val = r3(r1val, r2val)
            return r1val * F[tuple(A)] + r2val * F[tuple(B)] + r3val * F[tuple(C)]
        if isin_triangle(y, C, D, A):
            r1val = r1(y, C, D, A)
            r2val = r2(y, C, D, A)
            r3val = r3(r1val, r2val)
            return r1val * F[tuple(C)] + r2val * F[tuple(D)] + r3val * F[tuple(A)]

        return np.nan
