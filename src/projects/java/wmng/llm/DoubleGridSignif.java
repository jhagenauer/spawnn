package wmng.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import llm.LLMNG;
import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.GeoUtils;

public class DoubleGridSignif {

	private static Logger log = Logger.getLogger(DoubleGridSignif.class);

	public static void main(String[] args) {
		final Random r = new Random();
		int maxRun = 200;

		int threads = 4;
		final int[] fa = new int[] { 2 };
		final Dist<double[]> fDist = new EuclideanDist(fa);

		final int T_MAX = 120000;
		final int nrNeurons = 16; // je mehr neuronen, desto größer der unterschied?
		final double nbInit = (double) nrNeurons * 2.0 / 3.0;
		final double nbFinal = 1.0;
		final double lr1Init = 0.6;
		final double lr1Final = 0.01;
		final double lr2Init = 0.6;
		final double lr2Final = 0.01;

		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		for (int i = 0; i < maxRun; i++) {
			futures.add(es.submit(new Callable<double[]>() {

				@Override
				public double[] call() throws Exception {
					GridData data = new GridData(
							DoubleGrid2DUtils.createSpDepGrid(50, 50, true), 
							DoubleGrid2DUtils.createSpDepGrid(50, 50, true)
						);

					List<double[]> samplesTrain = data.samplesTrain;
					List<double[]> desiredTrain = data.desiredTrain;

					List<double[]> samplesVal = data.samplesVal;
					List<double[]> desiredVal = data.desiredVal;
					Map<double[], Map<double[], Double>> dMapVal = data.dMapVal;

					DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
					DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
					DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);

					double errorA, errorB;
					{ // WMNG + LLM
						double alpha = 0.7;
						double beta = 0.7;
						
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] rs = samplesTrain.get(r.nextInt(samplesTrain.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samplesTrain)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						Map<double[], Map<double[], Double>> dMapTrain = data.dMapTrain;
						SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);

						ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
						ng.useCtx = true;
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samplesTrain.size());
							ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
						}

						sorter.setWeightMatrix(dMapVal); // new weight-matrix

						bmuHist.clear();
						for (double[] d : samplesVal)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						// train histMap
						for (int i = 0; i < 100; i++) {
							List<double[]> rSamplesVal = new ArrayList<double[]>(samplesVal);
							Collections.shuffle(rSamplesVal);
							for (double[] x : rSamplesVal)
								sorter.sort(x, neurons);
						}

						List<double[]> responseVal = new ArrayList<double[]>();
						for (double[] x : samplesVal)
							responseVal.add(ng.present(x));
						errorA = getError(responseVal, desiredVal);
					}
					{
						List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, data.dMapTrain);

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						int[] nfa = new int[fa.length * 2];
						for (int i = 0; i < fa.length; i++) {
							nfa[i] = fa[i];
							nfa[i + fa.length] = fa[i] + samplesTrain.get(0).length;
						}

						Sorter<double[]> sorter = new DefaultSorter<>(new EuclideanDist(nfa));

						LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(lagedSamplesTrain.size());
							ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
						}

						List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(samplesVal, dMapVal);
						List<double[]> responseVal = new ArrayList<double[]>();
						for (double[] x : lagedSamplesVal)
							responseVal.add(ng.present(x));
						errorB = getError(responseVal, desiredVal);
					}
					return new double[]{errorA,errorB};
				}

			}));
		}
		es.shutdown();
		
		DescriptiveStatistics dsA = new DescriptiveStatistics();
		DescriptiveStatistics dsB = new DescriptiveStatistics();
		DescriptiveStatistics rmseDiff = new DescriptiveStatistics();
		for( Future<double[]> f : futures ) {
			try {
				double rmseA = f.get()[0];
				double rmseB = f.get()[1];
				dsA.addValue(rmseA);
				dsB.addValue(rmseB);
				
				rmseDiff.addValue( rmseA - rmseB );
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		log.debug("mean rmseA: "+dsA.getMean()+", mean rmseB: "+dsB.getMean() );
		
		double mean = rmseDiff.getMean();
		double se = rmseDiff.getStandardDeviation()/Math.sqrt(rmseDiff.getN());
		double tStatistic = mean/se;
		log.debug("mean: "+mean+", se: "+se+", t-Statistic: "+tStatistic);
		
		TDistribution tDist = new TDistribution(rmseDiff.getN()-1);
		log.debug("p-Value: "+tDist.cumulativeProbability(tStatistic));
		
		log.debug("CI: "+mean+"+/-"+Math.abs(tDist.inverseCumulativeProbability(0.025)*se));
		
		// http://www.statstutor.ac.uk/resources/uploaded/paired-t-test.pdf
		/*log.debug(new TDistribution(19).cumulativeProbability(-3.231));
		log.debug(new TDistribution(19).cumulativeProbability(3.231));

		log.debug(new TDistribution(19).cumulativeProbability(-1.729));
		log.debug(new TDistribution(19).cumulativeProbability(1.729));
		log.debug(new TDistribution(19).inverseCumulativeProbability(0.05));*/
	}
	
	public static double getError(List<double[]> response, List<double[]> desired ) {
		return Meuse.getRMSE(response, desired);
	}
}
