package cng_llm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;

public class SimulatedRegressionSingle {

	private static Logger log = Logger.getLogger(SimulatedRegressionSingle.class);

	enum method {
		error, y, attr
	};

	public static void main(String[] args) {

		final Random r = new Random();
		
		boolean firstWrite = true;
		
		final int ta = 2;
		final int[] fa = new int[] { 2 };
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		List<Object[]> params = new ArrayList<Object[]>();
		// best so far

		params.add(new Object[] { method.error, LLMNG.mode.martinetz, false, 40000, 10, 10, 2, 0.1, false, 1.0, 0.1, true, 0.1, 0.01 });		
		params.add(new Object[] { method.error, LLMNG.mode.fritzke,   false, 40000, 10, 10, 2, 0.01, false, 1.0, 0.1, true, 0.6, 0.001 });
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, false, 0.6, 0.01 });
		
		/*params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, false, 0.2, 0.1 });
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.01, true, 1.0, 0.01, false, 0.3, 0.01 });
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.01, true, 1.0, 0.01, true, 0.3, 0.01 });
		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, false, 0.2, 0.1 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, true, 0.2, 0.1 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.01, true, 1.0, 0.01, false, 0.2, 0.1 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, false, 0.2, 0.01 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.01, true, 1.0, 0.01, false, 0.2, 0.01 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.1, true, 1.0, 0.01, true, 0.2, 0.01 });		
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, false, 40000, 10, 10, 2, 0.01, true, 1.0, 0.01, true, 0.2, 0.01 });*/		
				
		for (Object[] p : params) {
			final int idx = params.indexOf(p);
			final method m = (method) (p[0]);
			final LLMNG.mode mode = (LLMNG.mode) p[1];
			final boolean ignSupport = (boolean) p[2];
			final int T_MAX = (int) p[3];
			final int nrNeurons = (int) p[4];
			final int L = (int) p[5];
			final int lInit = (int) p[6];
			final double lFinal = (double) p[7];
			final boolean lr1Power = (boolean) p[8];
			final double lr1Init = (double) p[9];
			final double lr1Final = (double) p[10];
			final boolean lr2Power = (boolean) p[11];
			final double lr2Init = (double) p[12];
			final double lr2Final = (double) p[13];

			long time = System.currentTimeMillis();
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 1; run++) {
				
				final List<double[]> samples = new ArrayList<double[]>();
				final List<double[]> desired = new ArrayList<double[]>();
				final Map<Integer, Set<double[]>> ref = new HashMap<Integer, Set<double[]>>();
				while (samples.size() < 20000) {
					double lon = r.nextDouble();
					double lat = r.nextDouble();
					double x = lon;
					int c = (int) Math.floor(x * 10); // class
					double coef = 2 * c / 10.0;
					double y = x * coef;

					double[] d = new double[] { lon, lat, x };
					samples.add(d);
					desired.add(new double[] { y });

					if (!ref.containsKey(c))
						ref.put(c, new HashSet<double[]>());
					ref.get(c).add(d);
				}

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							double[] n = Arrays.copyOf(d, d.length);
							neurons.add(n);
						}

						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> secSorter = null;
						if (m == method.error) {
							secSorter = new ErrorSorter(samples, desired);
						} else if (m == method.y) {
							secSorter = new DefaultSorter<>(new EuclideanDist(new int[] { ta }));
						} else if (m == method.attr) {
							secSorter = new DefaultSorter<>(fDist);
						}
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);

						DecayFunction nbRate = new PowerDecay((double) nrNeurons / lInit, lFinal);

						DecayFunction lrRate1;
						if (lr1Power)
							lrRate1 = new PowerDecay(lr1Init, lr1Final);
						else
							lrRate1 = new LinearDecay(lr1Init, lr1Final);

						DecayFunction lrRate2;
						if (lr2Power)
							lrRate2 = new PowerDecay(lr2Init, lr2Final);
						else
							lrRate2 = new LinearDecay(lr2Init, lr2Final);

						LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
						ng.aMode = mode;
						if (m == method.error)
							((ErrorSorter) secSorter).setLLMNG(ng);

						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}

						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));
						double mse = Meuse.getMSE(response, desired);

						Map<double[], Double> residuals = new HashMap<double[], Double>();
						for (int i = 0; i < response.size(); i++)
							residuals.put(samples.get(i), response.get(i)[0] - desired.get(i)[0]);

						Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);

						double coefError = 0;
						double xError = 0;
						for (double[] n : neurons) {
							double x = n[fa[0]];
							double coef = ng.matrix.get(n)[0][0];

							double desiredCoef = 2 * Math.floor(x * 10) / 10.0;
							coefError += Math.pow(coef - desiredCoef, 2);

							double desiredX = Math.floor(x) + 0.05;
							xError += Math.pow(x - desiredX, 2);
						}
						
						new DefaultSorter<>(fDist).sort(new double[]{0,0,0}, neurons);
						List<double[]> l = new ArrayList<double[]>();
						for( double[] n : neurons )
							l.add( new double[]{ n[0], ng.output.get(n)[0], ng.matrix.get(n)[0][0] } );
						DataUtils.writeCSV("output/neurons_"+idx+"_"+mode+".csv", l, new String[]{"x","out","coef"} );

						return new double[] { Math.sqrt(mse), ClusterValidation.getNormalizedMutualInformation(ref.values(), mapping.values()), coefError, xError };
					}
				}));
			}
			es.shutdown();

			DescriptiveStatistics ds[] = null;
			for (Future<double[]> ff : futures) {
				try {
					double[] ee = ff.get();
					if (ds == null) {
						ds = new DescriptiveStatistics[ee.length];
						for (int i = 0; i < ee.length; i++)
							ds[i] = new DescriptiveStatistics();
					}
					for (int i = 0; i < ee.length; i++)
						ds[i].addValue(ee[i]);
				} catch (InterruptedException ex) {
					ex.printStackTrace();
				} catch (ExecutionException ex) {
					ex.printStackTrace();
				}
			}

			log.debug((System.currentTimeMillis() - time) / 1000.0 + "s");

			try {
				String fn = "output/resultSynthetic2.csv";
				if (firstWrite) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("method,fritzkeMode,ignSupport,t_max,nrNeurons,l,lInit,lFinal,lr1Power,lr1Init,lr1Final,lr2Power,lr2Init,lr2Final,rmse,nmi,coefError,xError\n").getBytes());
				}
				String s = m + "," + mode + "," + ignSupport + "," + T_MAX + "," + nrNeurons + "," + L + "," + lInit + "," + lFinal + "," + lr1Power + "," + lr1Init + "," + lr1Final + "," + lr2Power + "," + lr2Init + "," + lr2Final + "";
				for (int i = 0; i < ds.length; i++)
					s += "," + ds[i].getMean();// +","+ds[i].getStandardDeviation();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public static DescriptiveStatistics getDS(List<Double> l) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (double d : l)
			ds.addValue(d);
		return ds;
	}

	public static double[] toArray(List<Double> l) {
		double[] r = new double[l.size()];
		for (int i = 0; i < l.size(); i++)
			r[i] = l.get(i);
		return r;
	}
}
