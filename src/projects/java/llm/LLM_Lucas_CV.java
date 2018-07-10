package llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class LLM_Lucas_CV {

	public enum function {
		Linear, Power
	};

	private static Logger log = Logger.getLogger(LLM_Lucas_CV.class);
	
	static double bestRMSE = Double.MAX_VALUE;
	static Object[] bestParam = null;

	public static void main(String[] args) {

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/lucas/lucas.shp"), true);
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, sdf.size());

		int[] fa = new int[] { 
				2, // TLA
				3, // beds
				9, // rooms
				10, // lotsize
				19, // age
		};
		int[] ga = new int[] { 20, 21 };
		int ta = 0;

		for (double[] d : sdf.samples) {
			d[19] = Math.pow(d[19], 2);
			d[10] = Math.log(d[10]);
			d[2] = Math.log(d[2]);
			d[1] = Math.log(d[1]);
		}

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<Object[]> params = new ArrayList<>();
		int t_max = 100000;
		int nrNeurons = 20;
		for( int l = 1; l <= 20; l++ )
		for (LLMNG.mode mode : new LLMNG.mode[] { LLMNG.mode.fritzke })
			for (double nb1Init : new double[] { nrNeurons, nrNeurons * 2.0 / 3, nrNeurons * 1.0/3 })
				for (double nb1Final : new double[] { 0.001, 0.01, 0.1, 1 })
					for (function nb1Func : new function[] { function.Power, function.Linear })
						for (double lr1Init : new double[] { 0.4, 0.6, 0.8, 1 })
							for (double lr1Final : new double[] { 0.001, 0.01, 0.1 })
								for (function lr1Func : new function[] { function.Power, function.Linear })
									for (double nb2Init : new double[] { nrNeurons, nrNeurons * 2.0 / 3, nrNeurons * 1.0/3  })
										for (double nb2Final : new double[] { 0.001, 0.01, 0.1, 1 })
											for (function nb2Func : new function[] { function.Power, function.Linear })
												for (double lr2Init : new double[] { 0.4, 0.6, 0.8, 1 })
													for (double lr2Final : new double[] { 0.001, 0.01, 0.1 })
														for (function lr2Func : new function[] { function.Power, function.Linear })
															params.add(new Object[] { 
																	t_max, nrNeurons, l, mode, 
																	nb1Init, nb1Final, nb1Func, 
																	lr1Init, lr1Final, lr1Func, 
																	nb2Init, nb2Final, nb2Func, 
																	lr2Init, lr2Final, lr2Func
																	});
		log.debug("Num params: "+params.size());
		Collections.shuffle(params);
		
		ExecutorService es = Executors.newFixedThreadPool(8);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		for ( Object[] param : params ) {
			final Object[] p = param;

			futures.add(es.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {

					int t_max = (int) p[0];
					int nrNeurons = (int) p[1];
					Sorter<double[]> sorter = new KangasSorter<>(gDist, fDist, (int)p[2]);
					LLMNG.mode mode = (LLMNG.mode) p[3];
					double nb1Init = (double) p[4];
					double nb1Final = (double) p[5];
					function nb1Func = (function) p[6];
					double lr1Init = (double) p[7];
					double lr1Final = (double) p[8];
					function lr1Func = (function) p[9];
					double nb2Init = (double) p[10];
					double nb2Final = (double) p[11];
					function nb2Func = (function) p[12];
					double lr2Init = (double) p[13];
					double lr2Final = (double) p[14];
					function lr2Func = (function) p[15];

					DescriptiveStatistics rmse = new DescriptiveStatistics();
					DescriptiveStatistics r2 = new DescriptiveStatistics();
					for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
						
						Random r1 = new Random(0);

						List<double[]> samplesTrain = new ArrayList<double[]>();
						for (int k : cvEntry.getKey()) {
							double[] d = sdf.samples.get(k);
							samplesTrain.add( Arrays.copyOf( d, d.length ) );
						}

						List<double[]> samplesTest = new ArrayList<double[]>();
						for (int k : cvEntry.getValue()) {
							double[] d = sdf.samples.get(k);
							samplesTest.add( Arrays.copyOf(d, d.length ) );
						}

						DataUtils.zScoreColumns( samplesTrain, samplesTest, fa );

						List<double[]> neurons = new ArrayList<>();
						while (neurons.size() < nrNeurons) {
							int idx = r1.nextInt(samplesTrain.size());
							double[] d = samplesTrain.get(idx);
							neurons.add( Arrays.copyOf(d,d.length) );
						}

						LLMNG llmng = new LLMNG(neurons, 
								getFunction(nb1Init, nb1Final, nb1Func),
								getFunction(lr1Init, lr1Final, lr1Func), 
								getFunction(nb2Init, nb2Final, nb2Func),
								getFunction(lr2Init, lr2Final, lr2Func), 
								sorter, fa, 1);
						llmng.aMode = mode;

						for (int t = 0; t < t_max; t++) {
							int j = r1.nextInt( samplesTrain.size() );
							double[] d = samplesTrain.get(j);
							llmng.train((double) t / t_max, d, new double[] { d[ta] });
						}

						List<double[]> response = new ArrayList<>();
						List<double[]> desired = new ArrayList<>();
						for (double[] d : samplesTest) {
							response.add( llmng.present( d ) );
							desired.add( new double[] { d[ta] } );
						}
						rmse.addValue( SupervisedUtils.getRMSE( response, desired ) );
						r2.addValue( SupervisedUtils.getR2( response, desired ) );
					}
					synchronized (this) {
						if( rmse.getMean() < bestRMSE ) {
							bestRMSE = rmse.getMean();
							bestParam = p;
							log.info(Arrays.toString(p)+", "+rmse.getMean()+", "+r2.getMean() );
						}
					}
					return new double[] { rmse.getMean(), r2.getMean() };
				}
			}));
		}
		es.shutdown();
		for( Future<double[]> f : futures ) {
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
	}

	public static DecayFunction getFunction(double init, double fin, function func) {
		if (func == function.Power)
			return new PowerDecay(init, fin);
		else if (func == function.Linear)
			return new LinearDecay(init, fin);
		else
			throw new RuntimeException("Unkown function");
	}
}
