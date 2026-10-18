// Kylie Tate
// 10-17-26
// Description: This is a small neural network in Java. This network has 4 -> 3 -> 2 structured
//              neural network, and is trained by manually implemented methods and classes.


public class Main {
    public static void main(String[] args) {
        // class variables
        double[][] W1 = { {-0.21, 0.72, -0.25, 1}, {-0.94, -0.41, -0.47, 0.63}, {0.15, 0.55, -0.49, -0.75}};
        double[][] B1 = { {0.1}, {-0.36}, {-0.31} };
        double[][] W2 = { {0.76, 0.48, -0.73}, {0.34, 0.89, -0.23} };
        double[][] B2 = { {0.16}, {-0.46} };
        double[][][] X = {{{0}, {1}, {0}, {1}}, {{1}, {0}, {1}, {0}}, {{0}, {0}, {1}, {1}}, {{1}, {1}, {0}, {0}}};
        double[][][] Y = {{{0}, {1}}, {{1}, {0}}};
        double[][][] ONE_VEC = {{{1}, {1}, {1}}, {{1}, {1}}};
        double[][] w1 = W1;
        double[][] b1 = B1;
        double[][] w2 = W2;
        double[][] b2 = B2;

        // instantiating other classes
        PrimaryOp op = new PrimaryOp();
        TrainRes[] r = new TrainRes[4];

        // three loops for each epoch, mini-batch, and input
        for (int f = 0; f < 6; f++) {
            System.out.println("\n##########################################################################################");
            System.out.println("Epoch #" + (f+1) + "\n");
            for (int s = 0; s < 2; s++) {
                System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
                System.out.println("Mini-Batch #" + (s+1) + "\n");
                for (int l = 0; l < 2; l++) {
                    // j chooses the input value for printing and func argument
                    int j = s * 2 + l;
                    System.out.println("************************************************");
                    System.out.println("Input X" + (j+1) + ":\n");
                    r[j] = op.example(w1, w2, b1, b2, X[j], Y[l], ONE_VEC); 
                    r[j].print();
                }
                // accounts for the pairs of inputs to determine the updated biases and weights
                int ind = s * 2;
                w1 = op.update(w1, r[ind].gw1, r[ind + 1].gw1);
                b1 = op.update(b1, r[ind].d1, r[ind + 1].d1);
                w2 = op.update(w2, r[ind].gw2, r[ind + 1].gw2);
                b2 = op.update(b2, r[ind].d2, r[ind + 1].d2);
                op.update_all(w1, w2, b1, b2);
            }
        }
    }
}