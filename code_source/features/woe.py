import numpy as np

class WOE:

    @staticmethod
    def quantile(x,y,df):
        q = np.linspace(0,1,21, endpoint=True)
        np.quantile(df[x],np.range(0,1,0.05))

        pass
