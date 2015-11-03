package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class SimulatedRegressionSingle {

	private static Logger log = Logger.getLogger(SimulatedRegressionSingle.class);

	enum method {
		error, y, attr
	};

	public static void main(String[] args) {

		final Random r = new Random();
		
		DataFrame df = DataUtils.readDataFrameFromCSV(new File("output/simulatedRegression.csv"), new int[]{}, true);
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		final Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
		for( double[] d : df.samples ) {
			double[] s = new double[]{ d[0] };
			samples.add( s );
			desired.add( new double[]{ d[1] } );
			int cl = (int)d[2];
			if( !ref.containsKey(cl) )
				ref.put( cl, new HashSet<double[]>() );
			ref.get(cl).add(s);
		}
		
		boolean firstWrite = true;
		
		final int ta = 1;
		final int[] fa = new int[] { 0 };

		final Dist<double[]> fDist = new EuclideanDist(fa);

		List<Object[]> params = new ArrayList<Object[]>();
		// best so far

		params.add(new Object[] { method.error, LLMNG.mode.martinetz, 8000000, 6, 2, 0.1, true, 0.6, 0.01, true, 0.6, 0.01 });		
		params.add(new Object[] { method.error, LLMNG.mode.fritzke,   8000000, 6, 2, 0.1, true, 0.6, 0.01, true, 0.6, 0.01 });
		params.add(new Object[] { method.error, LLMNG.mode.hagenauer, 8000000, 6, 2, 0.1, true, 0.6, 0.01, true, 0.6, 0.01 });
				
		for (Object[] p : params) {
			final int idx = params.indexOf(p);
			final method m = (method) (p[0]);
			final LLMNG.mode mode = (LLMNG.mode) p[1];
			final int T_MAX = (int) p[2];
			final int nrNeurons = (int) p[3];
			final int lInit = (int) p[4];
			final double lFinal = (double) p[5];
			final boolean lr1Power = (boolean) p[6];
			final double lr1Init = (double) p[7];
			final double lr1Final = (double) p[8];
			final boolean lr2Power = (boolean) p[9];
			final double lr2Init = (double) p[10];
			final double lr2Final = (double) p[11];

			long time = System.currentTimeMillis();
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 1; run++) {
			
				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							double[] n = Arrays.copyOf(d, d.length);
							neurons.add(n);
						}

						Sorter<double[]> sorter = null;
						if (m == method.error) {
							sorter = new ErrorSorter(samples, desired);
						} else if (m == method.y) {
							sorter = new DefaultSorter<>(new EuclideanDist(new int[] { ta }));
						} else if (m == method.attr) {
							sorter = new DefaultSorter<>(fDist);
						}
						
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
						ng.ignSupport = false;
						if (m == method.error)
							((ErrorSorter) sorter).setLLMNG(ng);

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
						
						new DefaultSorter<>(fDist).sort(new double[]{0,0,0}, neurons);
						List<double[]> l = new ArrayList<double[]>();
						for( double[] n : neurons )
							l.add( new double[]{ n[0], ng.output.get(n)[0], ng.matrix.get(n)[0][0] } );
						DataUtils.writeCSV("output/neurons_"+idx+"_"+mode+".csv", l, new String[]{"x","out","coef"} );
						/*double[] r = new double[l.size()*3];
						for( int i = 0; i < l.size(); i++ )
							for( int j = 0; j < l.get(i).length; j++ )
								r[i*3+j] = l.get(i)[j];
						return r;*/
						
						// TODO Kann dass zuordnung bzgl. der Abbilung beim Error-Sorter tw. nicht eindeutig ist?!
						List<double[]> lr = new ArrayList<double[]>();
						for( int i = 0; i < samples.size(); i++ ) {
							double[] d = samples.get(i);
							lr.add( new double[]{ d[0], desired.get(i)[0], ng.present(d)[0] } );
						}
						DataUtils.writeCSV("output/output_"+idx+"_"+mode+".csv", lr, new String[]{"x","desired","response"});
											
						return new double[] { Math.sqrt(mse), ClusterValidation.getNormalizedMutualInformation(ref.values(), mapping.values()) };
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
					Files.write(Paths.get(fn), ("method,fritzkeMode,t_max,nrNeurons,lInit,lFinal,lr1Power,lr1Init,lr1Final,lr2Power,lr2Init,lr2Final,rmse,nmi\n").getBytes());
				}
				String s = m + "," + mode + "," + T_MAX + "," + nrNeurons + "," + lInit + "," + lFinal + "," + lr1Power + "," + lr1Init + "," + lr1Final + "," + lr2Power + "," + lr2Init + "," + lr2Final + "";
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
