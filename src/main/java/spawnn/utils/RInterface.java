package spawnn.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.PowerDecay;

public class RInterface {

	public static int[] getCNGCluster(double[][] samples, int numNeurons, double nbStart, double nbEnd, double lrStart, double lrEnd, int[] ga, int[] fa, int l, int trainingTime) {
		Random r = new Random();

		Sorter<double[]> s = new KangasSorter<double[]>(new EuclideanDist(ga), new EuclideanDist(fa), l);
		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < numNeurons; i++) {
			double[] rs = samples[r.nextInt(samples.length)];
			neurons.add(Arrays.copyOf(rs, rs.length));
		}

		NG ng = new NG(neurons, new PowerDecay(nbStart, nbEnd), new PowerDecay(lrStart, lrEnd), s);
		for (int t = 0; t < trainingTime; t++) {
			double[] x = samples[r.nextInt(samples.length)];
			ng.train((double) t / trainingTime, x);
		}

		List<double[]> n = new ArrayList<double[]>(neurons);
		int[] re = new int[samples.length];
		for (int i = 0; i < samples.length; i++)
			re[i] = n.indexOf(s.getBMU(samples[i], neurons));
		return re;
	}
}
