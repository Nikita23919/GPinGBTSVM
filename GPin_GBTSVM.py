import numpy as np
from numpy import linalg
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def Twin_plane_1(X, d1, d2, tau1, tau2, eps1, eps2):
    A = X[X[:, -1] == 1, :-1]
    B = X[X[:, -1] != 1, :-1]
    C1 = A[:,:-1]
    C2 = B[:,:-1]
    R1 = A[:,-1]
    R2 = B[:,-1]
     
    g1 = A.shape[0]
    g2 = B.shape[0]
    e1 = np.ones((g1, 1))
    e2 = np.ones((g2, 1))
    R1 = R1.reshape(g1,1)
    R2 = R2.reshape(g2,1)
    
    P = np.hstack((C1, e1))
    Q = np.hstack((C2, e2))
    
    # 1st QPP
    PtP = np.dot(P.T, P) + d1 * np.eye(P.shape[1])   #P^TP+d_1I
    PtPQt = linalg.solve(PtP, Q.T)     #(P^TP+d_1I)^-1 Q^T
    QPtPQt = np.dot(Q, PtPQt)    # g2 * g2   Q(P^TP+d_1I)^-1 Q^T
    QPtPQt = (QPtPQt + QPtPQt.T) / 2
    #Objective function
    QPP11 = QPtPQt
    QPP1 = np.block([[QPP11,-QPP11], [-QPP11, QPP11]])   #quadratic part
    
    Q11 = (e2 + R2 - e2 * (eps1 / tau1))
    Q21 = -(e2 + R2 + e2 * (eps2 / tau2))
    Q1 = np.block([[Q11],[Q21]])   #linear part
    
    #Constraints
    G11 = (1 / tau1) * np.eye(g2)
    G12 = (1 / tau2) * np.eye(g2)
    G21 = -np.eye(g2)
    G1 = np.block([[G11, G12],[G21, G21]]) #LHS
    c11 = d2 * e2
    c21 = np.zeros((g2,1))
    c1 = np.block([[c11],[c21]])    #RHS
         
    alphaa1 = solvers.qp(matrix(QPP1),matrix(Q1,tc='d'),matrix(G1),matrix(c1,tc='d'))
    alphasol1 = np.array(alphaa1['x'])
    alpha1 = alphasol1[:g2]
    beta1 = alphasol1[g2:]
    z = -np.dot(PtPQt,alpha1-beta1)
    w1 = z[:len(z)-1]
    b1 = z[len(z)-1]
    return [w1,b1]

def Twin_plane_2(X, d3, d4, tau3, tau4, eps3, eps4):
    # 2nd QPP
    A = X[X[:, -1] == 1, :-1]
    B = X[X[:, -1] != 1, :-1]
    C1 = A[:,:-1]
    C2 = B[:,:-1]
    R1 = A[:,-1]
    R2 = B[:,-1]
     
    g1 = A.shape[0]
    g2 = B.shape[0]
    e1 = np.ones((g1, 1))
    e2 = np.ones((g2, 1))
    R1 = R1.reshape(g1,1)
    R2 = R2.reshape(g2,1)
    
    P = np.hstack((C1, e1))
    Q = np.hstack((C2, e2))
    QtQ = np.dot(Q.T, Q) + d3 * np.eye(Q.shape[1])
    QtQPt = linalg.solve(QtQ, P.T)
    PQtQPt = np.dot(P, QtQPt) #g1 * g1
    PQtQPt = (PQtQPt + PQtQPt.T) / 2
     #Objective function
    QPP21 = PQtQPt
    QPP2 = np.block([[QPP21,-QPP21], [-QPP21, QPP21]])    #quadratic part
    
    Q211 = e1 + R1 - (e1 * (eps3 / tau3)) 
    Q221 = -(e1 + R1 + (e1 * (eps4 / tau4)))
    Q2 = np.block([[Q211],[Q221]])           #linear part
    
    #Constraints
    G211 = (1 / tau3) * np.eye(g1)
    G212 = (1 / tau4) * np.eye(g1)
    G22 = -np.eye(g1)
    G2 = np.block([[G211, G212],[G22, G22]])   #LHS
    c211 = d4 * e1
    c221 = np.zeros((g1,1))
    c2 = np.block([[c211],[c221]])    #RHS
    
    alphaa2 = solvers.qp(matrix(QPP2),matrix(Q2,tc='d'),matrix(G2),matrix(c2,tc='d'))
    alphasol2 = np.array(alphaa2['x'])
    alpha2 = alphasol2[:g1]
    beta2 = alphasol2[g1:]
    z = np.dot(QtQPt,alpha2-beta2)
    w2 = z[:len(z)-1]
    b2 = z[len(z)-1]
    return [w2,b2]

class GPinGTSVM:
    def __init__(self, d1=1, d2=1, d3=1, d4=1, tau1=0.1, tau2=0.2, tau3=0.1, tau4=0.1, eps1=0.1, eps2=0.01, eps3=0.1, eps4=0.01):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.tau4 = tau4
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.eps4 = eps4
        
    def fit(self, X):        
        # Calculate plane parameters
        self.w1, self.b1 = Twin_plane_1(X, self.d1, self.d2, self.tau1, self.tau2, self.eps1, self.eps2)
        self.w2, self.b2 = Twin_plane_2(X, self.d3, self.d4, self.tau3, self.tau4, self.eps3, self.eps4)
        return self

    def predict(self, X):
        # Distance from the two planes
        dist1 = np.abs(np.dot(X, self.w1) + self.b1) #/ norm(self.w1)
        dist2 = np.abs(np.dot(X, self.w2) + self.b2) #/ norm(self.w2)

        # Predict class based on which plane the data point is closer to
        return np.where(dist1 < dist2, 1, -1)    
