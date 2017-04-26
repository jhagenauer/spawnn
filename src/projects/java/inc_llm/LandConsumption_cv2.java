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
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jfree.data.function.PowerFunction2D;

import com.vividsolutions.jts.geom.Point;

import chowClustering.LinearModel;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.ConstantDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class LandConsumption_cv2 {

	private static Logger log = Logger.getLogger(LandConsumption_cv2.class);

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

		List<double[]> samplesTrain = new ArrayList<>();
		List<double[]> desiredTrain = new ArrayList<>();
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();

		for (double[] d : samples) {
			if (r.nextDouble() < 0.8) {
				samplesTrain.add(d);
				desiredTrain.add(new double[] { d[ta] });
			} else {
				samplesVal.add(d);
				desiredVal.add(new double[] { d[ta] });
			}
		}

		boolean gaussian = true;
		boolean adaptive = true;

		int t_max = 1000000;
		int initNeurons = 4;

		double lrB = 0.001;
		double lrBln = 0.00001;
		double lrN = 0;
		double lrNln = 0;

		int aMax = t_max;
		int lambda = t_max;
		double alpha = 0.5;
		double beta = 0.000005;

		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < initNeurons; i++) {
			double[] d = samples.get(r.nextInt(samples.size()));
			neurons.add(Arrays.copyOf(d, d.length));
		}

		Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);

		IncLLM llm = new IncLLM(neurons, 
				new PowerDecay(0.1,0.0001), 
				new ConstantDecay(0), 
				new PowerDecay(0.01,0.0001),  
				new ConstantDecay(0), 
				sorter, aMax, lambda, alpha, beta, fa, 1, t_max);
		//IncLLM llm = new IncLLM(neurons, lrB, lrBln, lrN, lrNln, sorter, aMax, lambda, alpha, beta, fa, 1);
		int t = 0;
		for (; t < t_max; t++) {
			int idx = r.nextInt(samplesTrain.size());
			llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));
			
			if( t % 10000 == 0 ) {
				List<double[]> responseVal = new ArrayList<double[]>();
				for (int i = 0; i < samplesVal.size(); i++)
					responseVal.add(llm.present(samplesVal.get(i)));
				log.debug(t+" " + SupervisedUtils.getRMSE(responseVal, desiredVal));
				
				
				Map<double[],Set<double[]>> mTrain = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
				Map<double[],Set<double[]>> mVal = NGUtils.getBmuMapping(samplesVal, neurons, sorter);
				log.debug(t+" " + DataUtils.getMeanQuantizationError(mTrain, gDist) + "\t" + DataUtils.getMeanQuantizationError(mVal, gDist));
			}
		}

		List<double[]> responseVal = new ArrayList<double[]>();
		for (int i = 0; i < samplesVal.size(); i++)
			responseVal.add(llm.present(samplesVal.get(i)));

		log.debug("incLLM RMSE: " + SupervisedUtils.getRMSE(responseVal, desiredVal));

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
			log.debug("GWR: "+bw + "," + SupervisedUtils.getRMSE(predictions, samplesVal, ta));
		}
		
		// lm --------------
		
		LinearModel lm = new LinearModel(samplesTrain, fa, ta, false);
		List<Double> pred = lm.getPredictions(samplesVal, fa);
		log.debug("LM: "+SupervisedUtils.getRMSE(pred, samplesVal, ta));
		
		// k-means ------------
		Map<double[], Set<double[]>> mTrain = Clustering.kMeans(samplesTrain, initNeurons, gDist, 0.000001);
		Map<double[],Set<double[]>> mVal = new HashMap<>();
		for( double[] d : samplesVal ) {
			double[] bestK = null;
			for( double[] k : mTrain.keySet() )
				if( bestK == null || gDist.dist(d, k) < gDist.dist( d, bestK ) )
					bestK = k;
			if( !mVal.containsKey( bestK ))
				mVal.put(bestK, new HashSet<double[]>() );
			mVal.get(bestK).add(d);
		}
		log.debug("kMeans: " + DataUtils.getMeanQuantizationError(mTrain, gDist) + "\t" + DataUtils.getMeanQuantizationError(mVal, gDist));
	}
}
