package gwr.ga;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;

public class BuildTestData {

	private static Logger log = Logger.getLogger(BuildTestData.class);

	public static void main(String[] args) {
		Set<double[]> s = createSpDepTestData(16);
		DataUtils.writeCSV("output/spDat.csv", new ArrayList<double[]>(s), new String[] { "long", "lat", "beta", "x", "y" });
		log.debug(s.size());
	}

	public static Set<double[]> createSpDepTestData(int pointsPerRow ) {
		Random r = new Random(1000);
		Set<double[]> s = createSpDepTestData(1.0, 1.0, 0.5, true, 1.0 / pointsPerRow, r);

		double b_ = r.nextDouble() * 4 - 2;
		double x = r.nextDouble();
		s.add(new double[] { 0, 0, b_, x, b_ * x });
		return s;
	}

	private static Set<double[]> createSpDepTestData(double x_2, double y_2, double cut, boolean vert, double res, Random r) {
		Set<double[]> s = new HashSet<double[]>();
		// log.debug(x_2+","+y_2+","+cut+","+vert+","+res);

		double noiseSD = 0.05;
		double beta = r.nextDouble() * 4 - 2;
		double intercept = 0;
		if (vert) {
			for (double a = Math.ceil(cut / res) * res; a < x_2; a += res)
				for (double b = 0.0; b < y_2; b += res) {
					double x = r.nextDouble();
					double noise = r.nextGaussian() * noiseSD;
					s.add(new double[] { a, b, beta, x, intercept + beta * x + noise });
				}
			if (!s.isEmpty())
				s.addAll(createSpDepTestData(cut, y_2, cut, !vert, res, r));
		} else {
			for (double a = 0.0; a < x_2; a += res)
				for (double b = Math.ceil(cut / res) * res; b < y_2; b += res) {
					double x = r.nextDouble();
					double noise = r.nextGaussian() * noiseSD;
					s.add(new double[] { a, b, beta, x, intercept + beta * x + noise });
				}
			if (!s.isEmpty())
				s.addAll(createSpDepTestData(x_2, cut, cut / 2, !vert, res, r));
		}
		return s;
	}
}
