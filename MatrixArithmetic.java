// Kylie Tate
// 10-17-26
// Description: This is a small neural network in Java. This network has 4 -> 3 -> 2 structured
//              neural network, and is trained by manually implemented methods and classes.


// class containing basic matrix operations
public class MatrixArithmetic {

    // transposes a matrix
    public double[][] mat_transpose(double[][] a) {
        double[][] result = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) { 
                result[j][i] = a[i][j];
            }
        }        
        return result;
    }

    // add two matrices
    public double[][] mat_add(double[][] a, double[][] b) {
        if (a[0].length != b[0].length || a.length != b.length) { return null; }

        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) { 
                result[i][j] = a[i][j] + b[i][j];
            }
        }        
        return result;
    }

    // subtracts two matrices
    public double[][] mat_sub(double[][] a, double[][] b) {
        if (a[0].length != b[0].length || a.length != b.length) { return null; }
        return mat_add(a, mat_scale(b, -1));
    }

    // multiplies two matrices element by element
    public double[][] mat_haramard(double[][] a, double[][] b) {
        if (a[0].length != b[0].length || a.length != b.length) { return null; }

        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) { 
                result[i][j] = a[i][j] * b[i][j];
            }
        }        
        return result;
    }

    // calc matrix multiplication
    public double[][] mat_multi(double[][] a, double[][] b) {
        if (a[0].length != b.length) { return null; }

        double[][] result = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                double element = 0;
                for (int k = 0; k < b.length; k++) {
                    element += a[i][k] * b[k][j];
                }
                result[i][j] = element; 
            }
        }
        return result;
    }

    // scales a matrix by a constant
    double[][] mat_scale(double[][] a, double c) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                result[i][j] = a[i][j] * c;
            }
        }        
        return result;
    }

    // scales matrix by sigmoid to get activation values
    double[][] sigmoid(double[][] a) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                result[i][j] = 1/(1 + Math.exp(-a[i][j]));
            }
        }        
        return result;
    }

}
