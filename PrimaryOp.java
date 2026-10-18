// Kylie Tate
// 10-17-26
// Description: This is a small neural network in Java. This network has 4 -> 3 -> 2 structured
//              neural network, and is trained by manually implemented methods and classes.


// class containing neural network operations
public class PrimaryOp {
    MatrixArithmetic mat = new MatrixArithmetic();

    // calc forward pass (a)
    double[][] fwd_pass_eq(double[][] w, double[][] a, double[][] b) {
        return mat.sigmoid(mat.mat_add(mat.mat_multi(w, a), b));
    }

    // calc weight gradient
    double[][] w_grad(double[][] d, double[][] a) {
        return mat.mat_multi(d, mat.mat_transpose(a));
    }

    // calc delta/bias gradient for layer 2
    double[][] delta2(double[][] y, double[][] a2, double[][] one_vec) {
        return mat.mat_haramard(mat.mat_sub(a2, y), mat.mat_haramard(a2, mat.mat_sub(one_vec, a2)));
    }

    // calc delta/bias gradient for layer 1
    double[][] delta1(double[][] d2, double[][] w2, double[][] a1, double[][] one_vec) {
        return mat.mat_haramard(mat.mat_multi(mat.mat_transpose(w2), d2),mat.mat_haramard(a1, mat.mat_sub(one_vec, a1)));
    }
    
    // calc updated biases and gradients based off of the two inputs per mini-batch
    double[][] update(double[][] org_val, double[][] grad0, double[][] grad1) {
        return mat.mat_add(org_val, mat.mat_scale(mat.mat_add(grad0, grad1), -5));
    }

    // calc cost of the activation function
    double cost(double[][] a2, double[][] y) {
        double sum = 0;
        for (int i = 0; i < a2.length; i++) {
            sum += Math.pow(a2[i][0] - y[i][0], 2);
        }
        return 0.5*sum;
    }

    // defines each value and assigns to class variable in class TrainRes
    TrainRes example(double[][] w1, double[][] w2, double[][] b1, double[][] b2, double[][] x, double[][] y, double[][][] one_vec) {
        double[][] a1 = fwd_pass_eq(w1, x, b1);
        double[][] a2 = fwd_pass_eq(w2, a1, b2);
        double[][] d2 = delta2(y, a2, one_vec[1]); 
        double[][] d1 = delta1(d2, w2, a1, one_vec[0]); 
        double[][] gw2 = w_grad(d2, a1);
        double[][] gw1 = w_grad(d1, x);
        double cost = cost(a2, y);

        TrainRes res = new TrainRes();

        res.w1 = w1;
        res.b1 = b1;
        res.w2 = w2;
        res.b2 = b2;
        res.a1 = a1;
        res.a2 = a2;
        res.d2 = d2;
        res.d1 = d1;
        res.gw2 = gw2;
        res.gw1 = gw1;
        res.cost = cost;

        return res;
    }

    // prints updated biases and weights
    void update_all(double[][] w1, double[][] w2, double[][] b1, double[][] b2) {
        System.out.println("------------------------------------------\nRevised Biases and weights\n");
        print("b1", b1);
        print("w1", w1);
        print("b2", b2);
        print("w2", w2);
    }

    // formats each 2d double array to have a max of 6 decimal places in matrix form
    void print(String label, double[][] matrix) {
        System.out.println(label + ":");
        for (double[] row : matrix) {
            System.out.print("[");
            for (double val : row) {
                System.out.printf("%10.6f", val);
            }
            System.out.println("]");
        }
        System.out.println();
    }
}

// class containing general required values from forward passes and back propagation
class TrainRes {
    PrimaryOp op = new PrimaryOp();
    double[][] a1, a2;
    double[][] d1, d2;
    double[][] gw1, gw2;
    double[][] w1, w2;
    double[][] b1, b2;
    double cost;

    // prints each of the important values -> a1, a2, d1/gb1, d2/gb2, gw1, gw2, cost
    void print() {
        op.print("w1", w1);
        op.print("b1", b1);
        op.print("w2", w2);
        op.print("w1", w1);
        op.print("b2", b2);
        op.print("d1/gb1", d1);
        op.print("gw1", gw1);
        op.print("a2", a2);
        op.print("d2/gb2", d2);
        op.print("gw2", gw2);
        System.out.println("Cost: ");
        System.out.printf("%.6f", cost);
        System.out.println("\n");
    }
}
