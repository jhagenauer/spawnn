package rbf;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.RBF;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class RBFwAdaptWidth extends RBF {

	private static Logger log = Logger.getLogger(RBFwAdaptWidth.class);

	protected double error, beta, mod;
	protected double scale = 1;

	public RBFwAdaptWidth(Map<double[], Double> hidden, int out, Dist<double[]> dist, double delta, double beta, double mod) {
		super(hidden, out, dist, delta);
		this.beta = beta;
		this.mod = mod;
		this.error = 0;
	}

	@Override
	public void train(double[] x, double[] desired) {
		super.train(x, desired);

		// reduce error
		error = error - error * beta;

		// update prediction-error
		double[] response = present(x);
		double msq = 0;
		for (int i = 0; i < desired.length; i++)
			msq += Math.pow(desired[0] - response[0], 2) / desired.length;
		error += msq;
	}

	@Override
	public double[] present(double[] x) {
		double[] response = new double[weights.size()];
		for (double[] c : hidden.keySet()) {
			double output = Math.exp(-0.5 * Math.pow(dist.dist(x, c) / (scale * hidden.get(c)), 2));
			for (int i = 0; i < weights.size(); i++)
				response[i] += weights.get(i).get(c) * output;
		}
		return response;
	}

	public double getScale() {
		return scale;
	}

	protected double oldError = 0;

	public void adaptScale() {

		if (error > oldError) // change dir
			mod = -mod;

		scale += mod;
		oldError = error;
	}

	public static void main(String[] args) {
		final Random r = new Random();

		//List<double[]> all = DataUtils.readCSV("data/polynomial.csv", new int[] {});
		//DataUtils.zScore(all);
		SpatialDataFrame sd = DataUtils.readShapedata( new File("data/ontario/clipped/ontario_inorg_sel_final.shp"), new int[]{}, false);
		
		List<double[]> all = new ArrayList<double[]>();
		for( double[] d : sd.samples ) 
			all.add( new double[]{ d[2], d[3], Math.pow(d[33],1.0/2 ) } ); 
		int[] ga = new int[]{0,1};
		DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga) );
		DataUtils.zScoreColumns(all, new int[]{ 2 } );
		

		final Dist<double[]> dist = new EuclideanDist();

		final int maxK = 10;

		for (final double beta : new double[] { 0.00001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 }) {

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int j = 0; j < 25; j++) {

				Collections.shuffle(all);
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();

				for (double[] d : all) {
					//samples.add(new double[] { d[0], d[1], d[2], d[3], d[4] });
					//desired.add(new double[] { d[5] });
					
					samples.add(new double[] { d[0], d[1] });
					desired.add(new double[] { d[2] });
				}

				for (int k = 0; k < maxK; k++) {
					final int K = k;

					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {
							List<double[]> training = new ArrayList<double[]>();
							List<double[]> trainingDesired = new ArrayList<double[]>();
							List<double[]> validation = new ArrayList<double[]>();
							List<double[]> validationDesired = new ArrayList<double[]>();

							for (int i = 0; i < samples.size(); i++) {
								if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
									validation.add(samples.get(i));
									validationDesired.add(desired.get(i));
								} else {
									training.add(samples.get(i));
									trainingDesired.add(desired.get(i));
								}
							}

							Map<double[], Double> hidden = new HashMap<double[], Double>();

							Map<double[], Set<double[]>> clustering = Clustering.kMeans(samples, 11, dist);
							double qe = DataUtils.getMeanQuantizationError(clustering, dist);
							for (int i = 0; i < 25; i++) {
								Map<double[], Set<double[]>> tmp = Clustering.kMeans(samples, clustering.size(), dist);
								if (DataUtils.getMeanQuantizationError(tmp, dist) < qe) {
									qe = DataUtils.getMeanQuantizationError(tmp, dist);
									clustering = tmp;
								}
							}

							for (double[] c : clustering.keySet()) {
								double d = Double.MAX_VALUE;
								for (double[] n : clustering.keySet())
									if (c != n)
										d = Math.min(d, dist.dist(c, n));
								hidden.put(c, d);
							}

							RBFwAdaptWidth rbfwaw = new RBFwAdaptWidth(hidden, 1, dist, 0.05, beta, 0.05);

							for (int t = 0; t < 100000; t++) {
								int j = r.nextInt(samples.size());
								rbfwaw.train(samples.get(j), desired.get(j));

								if (t % 500 == 0)
									rbfwaw.adaptScale();
							}

							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : samples)
								response.add(rbfwaw.present(x));

							return new double[] { rbfwaw.getScale(), Meuse.getRMSE(response, desired), Math.pow(Meuse.getPearson(response, desired), 2) };
						}
					}));
				}
			}

			es.shutdown();
			DescriptiveStatistics[] ds = null;

			for (Future<double[]> f : futures) {
				try {
					double[] d = f.get();

					if (ds == null) {
						ds = new DescriptiveStatistics[d.length];
						for (int i = 0; i < d.length; i++)
							ds[i] = new DescriptiveStatistics();
					}

					for (int i = 0; i < d.length; i++)
						ds[i].addValue(d[i]);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}

			String desc = beta + "";

			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < ds.length; i++)
				sb.append(ds[i].getMean() + ",");
			log.debug(desc + "," + sb.substring(0, Math.min(sb.length(), 400)));
		}

	}
}
