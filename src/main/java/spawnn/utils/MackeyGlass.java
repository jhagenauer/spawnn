package spawnn.utils;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MackeyGlass {

	private double a;
	private double b;
	private double tau;
	private double x0;
	private double deltat;
	private Random r;

	public MackeyGlass(double a, double b, double tau, double x0, double deltat) {
		this.a = a;
		this.b = b;
		this.tau = tau;
		this.x0 = x0;
		this.deltat = deltat;
		this.r = new Random();
	}

	public double[] createSequence(int sample_n, double variation) {
		x0 += variation * (r.nextDouble() - 0.5);
		double time = 0.0;
		int index = 0;
		int historyLength = (int) Math.floor(tau / deltat);
		double[] xHistory = new double[historyLength];
		double x_t = x0;
		double[] X = new double[sample_n + 1];
		double[] T = new double[sample_n + 1];
		double x_t_minus_tau;
		double x_t_plus_deltat;
		for (int i = 0; i < sample_n + 1; i++) {
			X[i] = x_t;

			if (tau == 0.0) {
				x_t_minus_tau = 0.0;
			} else {
				x_t_minus_tau = xHistory[index];
			}

			x_t_plus_deltat = mackeyglassRK4(x_t, x_t_minus_tau, deltat, a, b);

			if (tau != 0) {
				xHistory[index] = x_t_plus_deltat;
				index = (index % (historyLength - 1)) + 1;
			}

			time = time + deltat;
			T[i] = time;
			x_t = x_t_plus_deltat;

		}
		return X;
	}
	
	private static double mackeyglassRK4(double x_t, double x_t_minus_tau, double deltat, double a, double b) {
		double k1 = deltat * mackeyglassEq(x_t, x_t_minus_tau, a, b);
		double k2 = deltat * mackeyglassEq(x_t + 0.5 * k1, x_t_minus_tau, a, b);
		double k3 = deltat * mackeyglassEq(x_t + 0.5 * k2, x_t_minus_tau, a, b);
		double k4 = deltat * mackeyglassEq(x_t + k3, x_t_minus_tau, a, b);
		return x_t + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;
	}

	private static double mackeyglassEq(double x_t, double x_t_minus_tau, double a, double b) {
		return (-b * x_t) + ((a * x_t_minus_tau) / (1 + Math.pow(x_t_minus_tau, 10.0)));
	}
	
	public static void main(String[] args) {

		double a = 0.2;
		double b = 0.1;
		double tau = 17;
		double x0 = 1.2; // init
		double deltat = 1.0; // frequenz

		MackeyGlass mg = new MackeyGlass(a, b, tau, x0, deltat);
		double[] seq = mg.createSequence(450000, 0.1);
		List<double[]> samples = new ArrayList<double[]>();
		for( double d : seq ) {
			//System.out.println(d);
			samples.add( new double[]{d});
		}
		
		try {
			DataUtils.writeCSV(new FileOutputStream("output/mg.csv"), samples);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
