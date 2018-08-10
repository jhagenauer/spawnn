package spawnn_toolkit;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.AugmentedDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.BmuGetterContext;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.bmu.SorterBmuGetter;
import spawnn.som.bmu.SorterBmuGetterContext;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.net.ContextSOM;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.GeoUtils;

public class PerformanceTests_NoThreads {

	private static Logger log = Logger.getLogger(PerformanceTests_NoThreads.class);

	public static void main(String[] args) {
		long time = System.currentTimeMillis();

		JDKRandomGenerator r = new JDKRandomGenerator();
		GaussianRandomGenerator grg = new GaussianRandomGenerator(r);

		int runs = 32;
		final int nrSamples = 10000;

		class TestPair {
			network n;
			contextModel m;
			double[] params;

			TestPair(network n, contextModel m, double[] params) {
				this.n = n;
				this.m = m;
				this.params = params;
			}
		}

		// Test c: alle, fixed param increases ga-dim
		// Test d: alle, fixed param increase fa-dim

		final int[] maxDim = new int[2];
		for (int i = 0; i < maxDim.length; i++)
			maxDim[i] = i + 1;

		// Test d, Grafik: x = dim, y = zeit
		log.debug("Test D");
		try {
			List<TestPair> models = new ArrayList<TestPair>();
			/*models.add(new TestPair(network.SOM, contextModel.Augmented, new double[] { 0.5 })); 
			models.add(new TestPair(network.SOM, contextModel.Weighted, new double[] { 0.5 })); 
			models.add(new TestPair(network.SOM, contextModel.GeoSOM, new double[] { 3 })); 
			models.add(new TestPair(network.SOM, contextModel.CNG, new double[] { 5 })); 
			models.add(new TestPair(network.SOM, contextModel.WMC, new double[] { 5, 0.5, 0.5 }));
			
			models.add(new TestPair(network.NG, contextModel.Augmented, new double[] { 0.5 }));
			models.add(new TestPair(network.NG, contextModel.Weighted, new double[] { 0.5 })); 
			models.add(new TestPair(network.NG, contextModel.CNG, new double[] { 5 })); 
			models.add(new TestPair(network.NG, contextModel.WMC, new double[] { 5, 0.5, 0.5 }));*/
			
			FileWriter fw = new FileWriter("output/testD.csv");
			fw.write("network,model,param,time\n");

			final int[] dim = new int[] { 6, 6 };

			for (int nrGDim : maxDim) {
				int nrFDim = 5;
				final List<double[]> samples = new ArrayList<double[]>();
				for (int i = 0; i < nrSamples; i++) {
					double[] d = new double[nrGDim + nrFDim];
					for (int j = 0; j < nrGDim; j++)
						d[j] = grg.nextNormalizedDouble();

					for (int j = 0; j < nrFDim; j++)
						d[j + nrGDim] = grg.nextNormalizedDouble();
					samples.add(d);
				}

				final int[] ga = new int[nrGDim];
				for (int i = 0; i < nrGDim; i++)
					ga[i] = i;
				final int[] fa = new int[nrFDim];
				for (int i = 0; i < nrFDim; i++)
					fa[i] = i + nrGDim;

				final Map<double[], List<double[]>> knns = GeoUtils.getKNNs(samples, new EuclideanDist(ga), 5, true);

				for (final TestPair entry : models) {

					List<Long> futures = new ArrayList<Long>();

					for (int run = 0; run < runs; run++) {
						System.gc();

						if (entry.n == network.SOM)
							futures.add(trainSOM(entry.m, entry.params, dim[0], dim[1], samples, knns, ga, fa));
						else
							futures.add(trainNG(entry.m, entry.params, dim[0] * dim[1], samples, knns, ga, fa));

					}

					DescriptiveStatistics ds = new DescriptiveStatistics();
					for (Long f : futures)
						ds.addValue(f);
					fw.write(entry.n + "," + entry.m + "," + nrGDim + "," + ds.getMean() + "\n");
				}
			}
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// Test c, Grafik: x = dim, y = zeit
		log.debug("Test C");
		try {

			final int[] dim = new int[] { 6, 6 };
			List<TestPair> models = new ArrayList<TestPair>();
			/*models.add(new TestPair(network.SOM, contextModel.Augmented, new double[] { 0.5 })); 
			models.add(new TestPair(network.SOM, contextModel.Weighted, new double[] { 0.5 })); 
			models.add(new TestPair(network.SOM, contextModel.GeoSOM, new double[] { 3 }));
			models.add(new TestPair(network.SOM, contextModel.CNG, new double[] { 5 }));
			models.add(new TestPair(network.SOM, contextModel.WMC, new double[] { 5, 0.5, 0.5 }));*/
			
			models.add(new TestPair(network.NG, contextModel.Augmented, new double[] { 0.5 })); 
			/*models.add(new TestPair(network.NG, contextModel.Weighted, new double[] { 0.5 })); 
			models.add(new TestPair(network.NG, contextModel.CNG, new double[] { 5 })); 
			models.add(new TestPair(network.NG, contextModel.WMC, new double[] { 5, 0.5, 0.5 }));*/
			
			FileWriter fw = new FileWriter("output/testC.csv");
			fw.write("network,model,param,time\n");

			for (int nrFDim : maxDim) {
				int nrGDim = 2;
				final List<double[]> samples = new ArrayList<double[]>();
				for (int i = 0; i < nrSamples; i++) {
					double[] d = new double[nrGDim + nrFDim];
					for (int j = 0; j < nrGDim; j++)
						d[j] = grg.nextNormalizedDouble();

					for (int j = 0; j < nrFDim; j++)
						d[j + nrGDim] = grg.nextNormalizedDouble();
					samples.add(d);
				}

				final int[] ga = new int[nrGDim];
				for (int i = 0; i < nrGDim; i++)
					ga[i] = i;
				final int[] fa = new int[nrFDim];
				for (int i = 0; i < nrFDim; i++)
					fa[i] = i + nrGDim;
				
				final Map<double[], List<double[]>> knns = GeoUtils.getKNNs(samples, new EuclideanDist(ga), 5, true);
				
				for (final TestPair entry : models) {
					List<Long> futures = new ArrayList<Long>();

					for (int run = 0; run < runs; run++) {
						System.gc();

						if (entry.n == network.SOM)
							futures.add(trainSOM(entry.m, entry.params, dim[0], dim[1], samples, knns, ga, fa));
						else
							futures.add(trainNG(entry.m, entry.params, dim[0] * dim[1], samples, knns, ga, fa));
					}

					DescriptiveStatistics ds = new DescriptiveStatistics();
					for (Long f : futures)
						ds.addValue(f);
					fw.write(entry.n + "," + entry.m + "," + nrFDim + "," + ds.getMean() + "\n");
				}
			}
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
				
		log.debug("took: " + ((System.currentTimeMillis() - time) / 1000.0 / 60.0) + " minutes.");
	}

	enum contextModel {
		GeoSOM, CNG, WMC, Weighted, Augmented
	};

	enum network {
		SOM, NG
	}

	public static long trainSOM(contextModel m, double[] params, int xDim, int yDim, List<double[]> samples, Map<double[], List<double[]>> knns, int[] ga, int[] fa) {
		Random r = new Random();

		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);

		int t_max = 100000;
		Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(xDim, yDim);
		SomUtils.initRandom(grid, samples);

		KernelFunction kf = new GaussKernel(new LinearDecay(10, 1));
		DecayFunction lr = new LinearDecay(1.0, 0.0);

		if (m == contextModel.WMC) {

			Map<double[], Map<double[], Double>> weights = getWeigthsFromKnns(knns, (int) params[0]);

			Map<GridPos, double[]> initNeurons = new HashMap<GridPos, double[]>();
			for (GridPos p : grid.getPositions()) {
				double[] d = grid.getPrototypeAt(p);
				double[] ns = Arrays.copyOf(d, d.length * 2);
				initNeurons.put(p, ns);
			}
			for (GridPos p : initNeurons.keySet())
				grid.setPrototypeAt(p, initNeurons.get(p));

			List<double[]> prototypes = new ArrayList<double[]>(grid.getPrototypes());
			Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
			for (double[] d : samples)
				bmuHist.put(d, prototypes.get(r.nextInt(prototypes.size())));

			SorterWMC s = new SorterWMC(bmuHist, weights, fDist, params[0], params[1]);
			BmuGetter<double[]> bg = new SorterBmuGetterContext(s);
			ContextSOM som = new ContextSOM(kf, lr, grid, (BmuGetterContext) bg, samples.get(0).length);

			long time = System.currentTimeMillis();
			s.setHistMutable(true);
			for (int t = 0; t < t_max; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / t_max, x);
			}
			s.setHistMutable(false);
			return System.currentTimeMillis() - time;

		} else {
			BmuGetter<double[]> bg = null;

			if (m == contextModel.CNG) {
				bg = new SorterBmuGetter<double[]>(new KangasSorter<double[]>(gDist, fDist, (int) params[0]));
			} else if (m == contextModel.GeoSOM) {
				bg = new KangasBmuGetter<double[]>(gDist, fDist, (int) params[0]);
			} else if (m == contextModel.Weighted) {
				double w = params[0];
				Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
				map.put(fDist, 1 - w);
				map.put(gDist, w);
				Dist<double[]> wDist = new WeightedDist<double[]>(map);
				bg = new DefaultBmuGetter<double[]>(wDist);
			} else if (m == contextModel.Augmented) {
				double a = params[0];
				Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
				bg = new DefaultBmuGetter<double[]>(aDist);
			} else {
				bg = new DefaultBmuGetter<double[]>(fDist);
			}

			SOM som = new SOM(kf, lr, grid, bg);

			long time = System.currentTimeMillis();
			for (int t = 0; t < t_max; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / t_max, x);
			}
			return System.currentTimeMillis() - time;
		}
	}

	private static Map<double[], Map<double[], Double>> getWeigthsFromKnns(Map<double[], List<double[]>> knns, int n) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();
		for (Entry<double[], List<double[]>> e : knns.entrySet())
			r.put(e.getKey(), e.getValue().subList(0, n));
		return GeoUtils.listsToWeightsOld(r);
	}

	public static long trainNG(contextModel m, double[] params, int nrNeurons, List<double[]> samples, Map<double[], List<double[]>> knns, int[] ga, int[] fa) {
		Random r = new Random();

		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);

		int t_max = 100000;
		if (m == contextModel.WMC) {
			Map<double[], Map<double[], Double>> weights = getWeigthsFromKnns(knns, (int) params[0]);

			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < nrNeurons; i++) {
				double[] rs = samples.get(r.nextInt(samples.size()));
				double[] d = Arrays.copyOf(rs, rs.length * 2);
				for (int j = rs.length; j < d.length; j++)
					d[j] = r.nextDouble();
				neurons.add(d);
			}

			Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
			for (double[] d : samples)
				bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

			Sorter<double[]> s = new SorterWMC(bmuHist, weights, fDist, params[0], params[1]);
			ContextNG ng = new ContextNG(neurons, 10, 0.01, 0.5, 0.005, (SorterWMC) s);

			long time = System.currentTimeMillis();
			for (int t = 0; t < t_max; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train((double) t / t_max, x);
			}
			return System.currentTimeMillis() - time;

		} else {
			Sorter<double[]> s;
			if (m == contextModel.CNG) {
				s = new KangasSorter<double[]>(gDist, fDist, (int) params[0]);
			} else if (m == contextModel.Weighted) {
				double w = params[0];
				Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
				map.put(fDist, 1 - w);
				map.put(gDist, w);
				Dist<double[]> wDist = new WeightedDist<double[]>(map);
				s = new DefaultSorter<double[]>(wDist);
			} else if (m == contextModel.Augmented) {
				double a = params[0];
				Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
				s = new DefaultSorter<double[]>(aDist);
			} else
				s = new DefaultSorter<double[]>(fDist);

			NG ng = new NG(nrNeurons, 10, 0.01, 0.5, 0.01, samples.get(0).length, s);

			long time = System.currentTimeMillis();
			for (int t = 0; t < t_max; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train((double) t / t_max, x);
			}
			return System.currentTimeMillis() - time;
		}
	}
}
