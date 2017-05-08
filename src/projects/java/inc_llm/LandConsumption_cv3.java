package inc_llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import com.vividsolutions.jts.geom.Point;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.ConstantDecay;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class LandConsumption_cv3 {

	private static Logger log = Logger.getLogger(LandConsumption_cv3.class);

	public static void main(String[] args) {
		Random r = new Random(0);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 52, 49, 10 };
		int ta = 7;
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);
		
		int t_max = 1000000;

		int aMax = 100;
		int lambda = 1000;
		double alpha = 0.5;
		double beta = 0.000005;
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 12, samples.size());
		
		SummaryStatistics ssIncLLM = new SummaryStatistics();
		SummaryStatistics ssIncLLM_LM = new SummaryStatistics();
		SummaryStatistics ssGWR = new SummaryStatistics();
		SummaryStatistics ssLM = new SummaryStatistics();
		SummaryStatistics sskMeansLM = new SummaryStatistics();
		
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(k));
				desiredTrain.add(new double[]{samples.get(k)[ta]});
			}
			
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) {
				samplesVal.add(samples.get(k));
				desiredVal.add(new double[]{samples.get(k)[ta]});
			}
			
			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < 2; i++) {
				double[] d = samples.get(r.nextInt(samples.size()));
				neurons.add(Arrays.copyOf(d, d.length));
			}

			Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);
			
			
			/*2017-05-08 07:49:17,502 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM: 9.411802702512432
			2017-05-08 07:49:17,502 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM_LM: 9.196530013012227
			2017-05-08 07:49:17,502 DEBUG [main] inc_llm.LandConsumption_cv3: GWR: 7.269787952781136
			2017-05-08 07:49:17,502 DEBUG [main] inc_llm.LandConsumption_cv3: LM: 11.482098798409755
			2017-05-08 07:49:17,502 DEBUG [main] inc_llm.LandConsumption_cv3: kMeans-LM: 8.941026905713562*/
			/*IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.005), 
					new ConstantDecay(0.005), 
					new LinearDecay(5.0E-4, 5.0E-6),  
					new LinearDecay(1.0E-5, 5.0E-6), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);*/
			
			/*2017-05-08 11:36:15,059 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM: 9.530422491288482
			2017-05-08 11:36:15,059 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM_LM: 9.25580271227574
			2017-05-08 11:36:15,059 DEBUG [main] inc_llm.LandConsumption_cv3: GWR: 7.269787952781136
			2017-05-08 11:36:15,059 DEBUG [main] inc_llm.LandConsumption_cv3: LM: 11.482098798409755
			2017-05-08 11:36:15,059 DEBUG [main] inc_llm.LandConsumption_cv3: kMeans-LM: 8.98633112230049*/
			/*IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.005), 
					new ConstantDecay(0.005), 
					new ConstantDecay(0.001),  
					new LinearDecay(1.0E-5, 5.0E-6), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);*/
			
			/*2017-05-08 11:52:43,015 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM: 1.346492143323006E148
			2017-05-08 11:52:43,015 DEBUG [main] inc_llm.LandConsumption_cv3: IncLLM_LM: 9.196069398637118
			2017-05-08 11:52:43,015 DEBUG [main] inc_llm.LandConsumption_cv3: GWR: 7.269787952781136
			2017-05-08 11:52:43,015 DEBUG [main] inc_llm.LandConsumption_cv3: LM: 11.482098798409755
			2017-05-08 11:52:43,015 DEBUG [main] inc_llm.LandConsumption_cv3: kMeans-LM: 8.99854390373231*/
			/*IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.005), 
					new PowerDecay(0.005,0.005), 
					new LinearDecay(0.001, 1.0E-4),  
					new PowerDecay(0.2,  5.0E-6), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);*/
			
			IncLLM llm = new IncLLM(neurons, 
					new ConstantDecay(0.01), 
					new ConstantDecay(0.005), 
					new LinearDecay(5.0E-4, 5.0E-6),  
					new LinearDecay(1.0E-5, 5.0E-6), 
					sorter, aMax, lambda, alpha, beta, fa, 1, t_max);
			
			for (int t = 0; t < t_max; t++) {
				int idx = r.nextInt(samplesTrain.size());
				llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));
			}

			List<double[]> responseVal = new ArrayList<double[]>();
			for (int i = 0; i < samplesVal.size(); i++)
				responseVal.add(llm.present(samplesVal.get(i)));

			ssIncLLM.addValue( SupervisedUtils.getRMSE(responseVal, desiredVal));
			
			{
				Map<double[],Set<double[]>> mTrain = NGUtils.getBmuMapping(samplesTrain, llm.neurons, sorter);
				Map<double[],Set<double[]>> mVal = NGUtils.getBmuMapping(samplesVal, llm.neurons, sorter);
						
				List<Set<double[]>> lTrain = new ArrayList<>();
				List<Set<double[]>> lVal = new ArrayList<>();
				for( double[] d : mTrain.keySet() ) { 
					lTrain.add(mTrain.get(d));
					lVal.add(mVal.get(d));
				}			
				LinearModel lm = new LinearModel( samplesTrain, lTrain, fa, ta, false);
				List<Double> pred = lm.getPredictions(samplesVal, lVal, fa);
				ssIncLLM_LM.addValue( SupervisedUtils.getRMSE(pred, samplesVal, ta));
			}
			
			boolean gaussian = true;
			boolean adaptive = true;
			for (double bw : new double[]{ 8 }) {

				Map<double[], Double> bandwidth = new HashMap<>();
				for (double[] a : samples) {
					if (!adaptive)
						bandwidth.put(a, bw);
					else {
						int k = (int) bw;
						List<double[]> s = new ArrayList<>(samples);
						Collections.sort(s, new Comparator<double[]>() {
							@Override
							public int compare(double[] o1, double[] o2) {
								return Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
							}
						});
						bandwidth.put(a, gDist.dist(s.get(k - 1), a));
					}
				}

				DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samplesTrain, ta));
				DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samplesTrain, fa, true));

				DoubleMatrix XVal = new DoubleMatrix(LinearModel.getX(samplesVal, fa, true));
				List<Double> predictions = new ArrayList<>();
				for (int i = 0; i < samplesVal.size(); i++) {
					double[] a = samplesVal.get(i);

					DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
					for (int j = 0; j < X.getRows(); j++) {
						double[] b = samplesTrain.get(j);
						double d = gDist.dist(a, b);

						double w;
						if (gaussian) // Gaussian
							w = Math.exp(-0.5 * Math.pow(d / bandwidth.get(a), 2));
						else // bisquare
							w = Math.pow(1.0 - Math.pow(d / bandwidth.get(a), 2), 2);
						XtW.putColumn(j, X.getRow(j).mul(w));
					}
					DoubleMatrix XtWX = XtW.mmul(X);
					DoubleMatrix beta2 = Solve.solve(XtWX, XtW.mmul(Y));

					predictions.add(XVal.getRow(i).mmul(beta2).get(0));
				}
				ssGWR.addValue(SupervisedUtils.getRMSE(predictions, samplesVal, ta));
			}
			
			{
				LinearModel lm = new LinearModel(samplesTrain, fa, ta, false);
				List<Double> pred = lm.getPredictions(samplesVal, fa);
				ssLM.addValue(SupervisedUtils.getRMSE(pred, samplesVal, ta));
			}
			
			
			{
				Map<double[], Set<double[]>> mTrain = Clustering.kMeans(samplesTrain, llm.neurons.size(), gDist, 0.000001);
				Map<double[],Set<double[]>> mVal = new HashMap<>(); // assign val to k-centroids
				for( double[] k : mTrain.keySet() )
					mVal.put( k, new HashSet<>() );				
				for( double[] d : samplesVal ) {
					double[] bestK = null;
					for( double[] k : mTrain.keySet() )
						if( bestK == null || gDist.dist(d, k) < gDist.dist( d, bestK ) )
							bestK = k;
					mVal.get(bestK).add(d);
				}
						
				List<Set<double[]>> lTrain = new ArrayList<>();
				List<Set<double[]>> lVal = new ArrayList<>();
				for( double[] d : mTrain.keySet() ) { 
					lTrain.add(mTrain.get(d));
					lVal.add(mVal.get(d));
				}			
				LinearModel lm = new LinearModel( samplesTrain, lTrain, fa, ta, false);
				List<Double> pred = lm.getPredictions(samplesVal, lVal, fa);
				sskMeansLM.addValue(SupervisedUtils.getRMSE(pred, samplesVal, ta));
			}
		}
		
		log.debug("IncLLM: "+ssIncLLM.getMean());
		log.debug("IncLLM_LM: "+ssIncLLM_LM.getMean());
		log.debug("GWR: "+ssGWR.getMean());
		log.debug("LM: "+ssLM.getMean());
		log.debug("kMeans-LM: "+sskMeansLM.getMean());
	}
}
