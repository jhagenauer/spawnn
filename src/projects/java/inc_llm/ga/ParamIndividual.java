package inc_llm.ga;

import java.io.File;
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

import inc_llm.IncLLM;
import inc_llm.OptimizeIncLLM_CV.parNames;
import rbf.Meuse;
import regionalization.ga.GAIndividual;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class ParamIndividual implements GAIndividual {
	

	static final List<double[]> samples, desired;
	static final int[] fa;
	static final Dist<double[]> gDist, fDist;
	
	static {
		samples = new ArrayList<double[]>();
		desired = new ArrayList<double[]>();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), new int[] { 0, 1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length - 1);

			samples.add(nd);
			desired.add(new double[] { d[d.length - 1] });
		}
		
		 fa = new int[samples.get(0).length - 2]; // omit geo-vars
			for (int i = 0; i < fa.length; i++)
				fa[i] = i + 2;
			final int[] ga = new int[] { 0, 1 };

			gDist = new EuclideanDist(ga);
			fDist = new EuclideanDist(fa);
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
	}

	final Random r = new Random();
	
	
	
	static final double[][] vals = new double[][]{
		new double[] { 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 }, // lrB
		new double[] { 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 }, // lrBln
		new double[] { 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0 }, // lrN
		new double[] { 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0 }, // lrNln
		new double[] { 60, 80, 100, 140, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800 }, // aMax
		new double[] { 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 280, 320, 360, 720, 1080, 1420 }, // lambda
		new double[] { 0.45, 0.5, 0.55 }, // alpha
		new double[] { 0.0005, 0.00005, 0.000005, 0.0000005, 0.0 } // beta
	};
	
	public String paramsToString() {
		String s = "";
		for( int i = 0; i < vals.length; i++ )
			s+= vals[i][chromosome[i]] +",";
		return s;
	}

	private int[] chromosome = new int[vals.length];
	
	public int[] getChromosome() {
		return chromosome;
	}

	public ParamIndividual() {
		for( int i = 0; i < vals.length; i++ )
			chromosome[i] = r.nextInt(vals[i].length);
	}
	
	public ParamIndividual(int[] chromosome ) {
		this.chromosome = chromosome;
	}

	@Override
	public int compareTo(GAIndividual gai) {
		return Double.compare(getValue(), gai.getValue());
	}

	@Override
	public GAIndividual mutate() {
		int[] mc = Arrays.copyOf(chromosome, chromosome.length);
		for( int i = 0; i < mc.length; i++ ) {
			if( r.nextDouble() < 1.0/mc.length )
				mc[i] = r.nextInt(vals[i].length);
		}
		return new ParamIndividual(mc);
	}

	@Override
	public GAIndividual recombine(GAIndividual mother) {
		int[] mcA = Arrays.copyOf(chromosome, chromosome.length);
		int[] mcB = ((ParamIndividual)mother).getChromosome();
		for( int i = 0; i < mcA.length; i+=2 )
			mcA[i] = mcB[i];
		return new ParamIndividual(mcA);
	}
	
	double value = -1;

	public static int numVal = 128;
	public double getValue() {
		
		if( value > 0 )
			return value;
		
		final Map<parNames, Object> pa = new HashMap<parNames, Object>();
		pa.put(parNames.tMax, 40000);
		pa.put(parNames.initNeurons, 2);
		pa.put(parNames.neuronsMax, 25);
		pa.put(parNames.lrB, vals[0][chromosome[0]]);
		pa.put(parNames.lrBln, vals[1][chromosome[1]]);
		pa.put(parNames.lrN, vals[2][chromosome[2]]);
		pa.put(parNames.lrNln, vals[3][chromosome[3]]);
		pa.put(parNames.aMax,vals[4][chromosome[4]]);
		pa.put(parNames.lambda, vals[5][chromosome[5]]);
		pa.put(parNames.alpha, vals[6][chromosome[6]]);
		pa.put(parNames.beta, vals[7][chromosome[7]]);
									
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < numVal; run++) {

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

						int t_max = (Integer) pa.get(parNames.tMax);
						int initNeurons = (Integer) pa.get(parNames.initNeurons);
						int neuronsMax = (Integer) pa.get(parNames.neuronsMax);
						double lrB = (Double) pa.get(parNames.lrB);
						double lrBln = (Double) pa.get(parNames.lrBln);
						double lrN = (Double) pa.get(parNames.lrN);
						double lrNln = (Double) pa.get(parNames.lrBln);
						int aMax = ((Double) pa.get(parNames.aMax)).intValue();
						int lambda = ((Double) pa.get(parNames.lambda)).intValue();
						double alpha = (Double) pa.get(parNames.alpha);
						double beta = (Double) pa.get(parNames.beta);

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < initNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);

						IncLLM llm = new IncLLM(neurons, lrB, lrBln, lrN, lrNln, sorter, aMax, lambda, alpha, beta, fa, 1);
						llm.maxNeurons = neuronsMax;
						int t = 0;
						for (; t < t_max; t++) {
							int idx = r.nextInt(samplesTrain.size());
							llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));

							/*if (llm.getNeurons().size() >= neuronsMax)
								break;*/
						}

						List<double[]> responseVal = new ArrayList<double[]>();
						for (int i = 0; i < samplesVal.size(); i++)
							responseVal.add(llm.present(samplesVal.get(i)));

						return new double[] { Meuse.getRMSE(responseVal, desiredVal) };
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
			value = ds[0].getMean();
			return value;
		}
}
