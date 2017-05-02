package inc_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class OptimizeIncLLM_CV {

	private static Logger log = Logger.getLogger(OptimizeIncLLM_CV.class);

	enum method {
		incLLM
	};
	
	public enum parNames {
		tMax, initNeurons, neuronsMax, lrB, lrBln, lrN, lrNln, aMax, lambda, alpha, beta
	};

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), new int[] { 0, 1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length - 1);

			samples.add(nd);
			desired.add(new double[] { d[d.length - 1] });
		}

		final int[] fa = new int[samples.get(0).length - 2]; // omit geo-vars
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
		/* DataUtils.zScoreGeoColumns(samples, ga, gDist); //not necessary */

		// ------------------------------------------------------------------------

		List<Map<parNames,Object>> params = new ArrayList<Map<parNames,Object>>();
		
		for( double lrB : new double[]{ 0.2, 0.1, 0.05, 0.01 } )
		for( double lrBln : new double[]{ 0.2, 0.1, 0.05, 0.01 } )
		for( double lrN : new double[]{ 0.2, 0.1, 0.05, 0.01} )
		for( double lrNln : new double[]{ 0.2, 0.1, 0.05, 0.01 } )
		for( int aMax : new int[]{100, 200, 300, 400, 500, 600, 700} )
		for( int lambda : new int[]{40, 60, 80, 100, 120, 140 } )
		for( double alpha : new double[]{ 0.5, 0.45, 0.55 } )
		for( double beta : new double[]{ 0.00005, 0.000005 } ) {
		Map<parNames,Object> p = new HashMap<parNames,Object>();
		p.put(parNames.tMax, 40000);
		p.put(parNames.initNeurons, 2);
		p.put(parNames.neuronsMax, 25);
		p.put(parNames.lrB, lrB);
		p.put(parNames.lrBln, lrBln);
		p.put(parNames.lrN, lrN);
		p.put(parNames.lrNln,lrNln);
		p.put(parNames.aMax, aMax);
		p.put(parNames.lambda, lambda);
		p.put(parNames.alpha, alpha);
		p.put(parNames.beta, beta);
		params.add(p);
		}
		
		for ( final Map<parNames,Object> pa: params) {

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < 8; run++) {

					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {
							
							List<double[]> samplesTrain = new ArrayList<double[]>(samples);
							List<double[]> desiredTrain = new ArrayList<double[]>(desired);
							List<double[]> samplesVal = new ArrayList<double[]>();
							List<double[]> desiredVal = new ArrayList<double[]>();

							while (samplesVal.size() < samples.size() * 0.7) {
								int idx = r.nextInt(samplesTrain.size());
								samplesVal.add(samplesTrain.remove(idx));
								desiredVal.add(desiredTrain.remove(idx));
							}

							int t_max = (Integer)pa.get(parNames.tMax);
							int initNeurons = (Integer)pa.get(parNames.initNeurons);
							int neuronsMax = (Integer)pa.get(parNames.neuronsMax);
							double lrB = (Double)pa.get(parNames.lrB);
							double lrBln = (Double)pa.get(parNames.lrBln);
							double lrN = (Double)pa.get(parNames.lrN);
							double lrNln = (Double)pa.get(parNames.lrBln);
							int aMax = (Integer)pa.get(parNames.aMax);
							int lambda = (Integer)pa.get(parNames.lambda);
							double alpha = (Double)pa.get(parNames.alpha);
							double beta = (Double)pa.get(parNames.beta);
														
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < initNeurons; i++) {
								double[] d = samples.get(r.nextInt(samples.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
						
							Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);
							
							IncLLM llm = new IncLLM(neurons, lrB, lrBln, lrN, lrNln, sorter, aMax, lambda, alpha, beta, fa, 1);
							int t = 0;
							for (; t < t_max; t++) {
								int idx = r.nextInt(samplesTrain.size());
								llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx) );
								
								if( llm.getNeurons().size() >= neuronsMax )
									break;
							}
							
							List<double[]> responseVal = new ArrayList<double[]>();
							for( int i = 0; i < samplesVal.size(); i++ )
								responseVal.add( llm.present(samplesVal.get(i)));
							
							return new double[] {
									t,
									llm.getNeurons().size(),
									0,//llm.getConections().size(),
									Meuse.getRMSE(responseVal, desiredVal),
									Meuse.getR2(responseVal, desiredVal)
									};
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

				try {
					String fn = "output/resultIncLLM_CV.csv";
					if (firstWrite) {
						firstWrite = false;
						String s = "";
						for( parNames pn : parNames.values() )
							s += pn +",";
						s += "t,neurons,connections,rmse,r2\n";
						Files.write(Paths.get(fn), s.getBytes());
					}
					String s = "";
					for( parNames pn : parNames.values() )
						s += pa.get(pn)+",";
					for (int i = 0; i < ds.length; i++)
						s += ds[i].getMean()+",";
					s += "\n";
					Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
					System.out.print(s);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
