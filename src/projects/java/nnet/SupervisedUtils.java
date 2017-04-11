package nnet;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

public class SupervisedUtils {

	private static Logger log = Logger.getLogger(SupervisedUtils.class);

	public static List<Entry<List<Integer>, List<Integer>>> getCVList(int numFolds, int numRepeats, int numSamples) {
		Random r = new Random(0);
		
		List<Entry<List<Integer>, List<Integer>>> cvList = new ArrayList<Entry<List<Integer>, List<Integer>>>();
		for (int repeat = 0; repeat < numRepeats; repeat++) {
			List<Integer> l = new ArrayList<Integer>();
			for (int i = 0; i < numSamples; i++)
				l.add(i);
			Collections.shuffle(l,r);
			if (numFolds == 0) { // full
				List<Integer> train = new ArrayList<Integer>(l);
				List<Integer> val = new ArrayList<Integer>(l);
				cvList.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train, val));
			} else { // n-fold cv
				int foldSize = numSamples / numFolds;
				for (int fold = 0; fold < numFolds; fold++) {
					List<Integer> val = new ArrayList<Integer>(l.subList(fold * foldSize, (fold + 1) * foldSize));
					List<Integer> train = new ArrayList<Integer>( l.subList(0, (fold * foldSize) ) );
					train.addAll( l.subList( (fold + 1) * foldSize, numSamples ) );
					cvList.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train, val));
				}
			}
		}
		return cvList;
	}

	// Maybe we should remove this function due to triviality
	@Deprecated
	public static double getRMSE(List<double[]> response, List<double[]> desired) {
		return Math.sqrt(getMSE(response, desired));
	}
	
	public static double getRMSE(List<Double> response, List<double[]> samples, int ta ) {
		return Math.sqrt(getMSE(response, samples, ta));
	}

	// Mean sum of squares
	@Deprecated
	public static double getMSE(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException("response.size() != desired.size()");

		double mse = 0;
		for (int i = 0; i < response.size(); i++)
			mse += Math.pow(response.get(i)[0] - desired.get(i)[0], 2);
		return mse / response.size();
	}
	
	public static double getMSE(List<Double> response, List<double[]> samples, int ta) {
		return getResidualSumOfSquares(response, samples, ta) / response.size();
	}

	@Deprecated
	public static double getR2(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException();

		double ssRes = 0;
		for (int i = 0; i < response.size(); i++)
			ssRes += Math.pow(desired.get(i)[0] - response.get(i)[0], 2);
		
		SummaryStatistics ss = new SummaryStatistics();
		for ( double[] d : desired )
			ss.addValue(d[0]);

		double mean = 0;
		for (double[] d : desired)
			mean += d[0];
		mean /= desired.size();

		double ssTot = 0;
		for (double[] d : desired )
			ssTot += Math.pow(d[0] - mean, 2);
		
		return 1.0 - ssRes / ssTot;
	}
	
	public static double getR2(List<Double> response, List<double[]> samples, int ta) {
		if (response.size() != samples.size())
			throw new RuntimeException("response size != samples size ("+response.size()+"!="+samples.size()+")" );

		double ssRes = 0;
		for (int i = 0; i < response.size(); i++)
			ssRes += Math.pow(samples.get(i)[ta] - response.get(i), 2);
		
		SummaryStatistics ss = new SummaryStatistics();
		for ( double[] d : samples )
			ss.addValue(d[ta]);

		double mean = 0;
		for (double[] d : samples)
			mean += d[ta];
		mean /= samples.size();

		double ssTot = 0;
		for (double[] d : samples )
			ssTot += Math.pow(d[ta] - mean, 2);
		
		return 1.0 - ssRes / ssTot;
	}

	public static double getPearson(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException();

		double meanDesired = 0;
		for (double[] d : desired)
			meanDesired += d[0];
		meanDesired /= desired.size();

		double meanResponse = 0;
		for (double[] d : response)
			meanResponse += d[0];
		meanResponse /= response.size();

		double a = 0;
		for (int i = 0; i < response.size(); i++)
			a += (response.get(i)[0] - meanResponse) * (desired.get(i)[0] - meanDesired);

		double b = 0;
		for (int i = 0; i < response.size(); i++)
			b += Math.pow(response.get(i)[0] - meanResponse, 2);
		b = Math.sqrt(b);

		double c = 0;
		for (int i = 0; i < desired.size(); i++)
			c += Math.pow(desired.get(i)[0] - meanDesired, 2);
		c = Math.sqrt(c);

		if (b == 0 || c == 0) // not sure about if this is ok
			return 0;

		return a / (b * c);
	}
	
	public static double getMultiLogLoss(List<Double> response, List<double[]> samples, int ta ) {
		List<double[]> desired = new ArrayList<double[]>();
		List<double[]> resp = new ArrayList<double[]>();
		for( int i = 0; i < samples.size(); i++ ) {
			resp.add( new double[]{ response.get(i) } );
			desired.add( new double[] { samples.get(i)[ta] } );
		}
		return getMultiLogLoss(resp, desired);
	}
	
	public static double getMultiLogLoss(List<double[]> response, List<double[]> desired ) {
		 double eps = Math.pow(10, -15);
		 double ll = 0;
		 for( int i = 0; i < response.size(); i++ ) 
			 for( int j = 0; j < response.get(i).length; j++ ) {
				 double a = desired.get(i)[j];
				 double p = Math.min(Math.max(eps, response.get(i)[j]), 1.0-eps);
				 ll += a*Math.log(p) + (1.0-a)*Math.log(1.0-p);
			 }
		 return ll * -1.0/desired.size();
	}

	public static double getAIC(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + 2 * nrParams;
	}
	
	public static double getAICc(double mse, double nrParams, int nrSamples) {
		if( nrSamples - nrParams - 1 <= 0 ) {
			log.error(nrSamples+","+nrParams);
			System.exit(1);
		}
		return getAIC(mse, nrParams, nrSamples) + (2.0 * nrParams * (nrParams + 1)) / (nrSamples - nrParams - 1);
	}
	
	// I don't know why, but that's how it is done in the GWMODEL package
	// dp.n*log(sigma.hat2) + dp.n*log(2*pi) +dp.n+tr.S
	public static double getAIC_GWMODEL(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + nrSamples * Math.log(2*Math.PI) + nrSamples + nrParams;
		
	}
	
	// I don't know why, but that's how it is done in the GWMODEL package
	// ##AICc = 	dev + 2.0 * (double)N * ( (double)MGlobal + 1.0) / ((double)N - (double)MGlobal - 2.0);
	// lm_AICc= dp.n*log(lm_RSS/dp.n)+dp.n*log(2*pi)+dp.n+2*dp.n*(var.n+1)/(dp.n-var.n-2)
	public static double getAICc_GWMODEL(double mse, double nrParams, int nrSamples) {
		if( nrSamples - nrParams - 2 <= 0 ) 
			throw new RuntimeException("too few samples! "+nrSamples+" "+nrParams);
		return nrSamples * ( Math.log(mse) + Math.log(2*Math.PI) + 1 ) + (2.0 * nrSamples * ( nrParams + 1 ) ) / (nrSamples - nrParams - 2);
	}

	public static double getBIC(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + nrParams * Math.log(nrSamples);
	}
	
	// TODO Check, not sure if correct
	public static double getAUC(List<double[]> props, List<double[]> desired ) {
		List<double[]> l = new ArrayList<>(props);
		Collections.sort(l, new Comparator<double[]>() {
			@Override
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[0], o2[0]);
			}			
		});
		
		int posTotal = 0;
		for( double[] d : desired )
			if( d[0] == 1 )
				posTotal++;
		
		double auc = 0;
		double preP = 0;
		for( int i = 0; i < l.size(); i++ ) {
			double p = l.get(i)[0]; // <= p  is 0
			
			int posCor = 0;
			for( int j = i; j <= desired.size(); j++ )
				if( desired.get(j)[0] == 1 )
					posCor++;
			auc += (p-preP) * (double)posCor/posTotal; // width * height
			preP = p;
		}
		return auc;
	}
	
	public static void main(String[] args) {
		List<Entry<List<Integer>, List<Integer>>> cvList = getCVList(2,1,10);
		System.out.println(cvList);;
	}

	public static double getResidualSumOfSquares(List<Double> response, List<double[]> samples, int ta) {
		if (response.size() != samples.size())
			throw new RuntimeException("response.size() != samples.size() "+response.size()+","+samples.size());

		double rss = 0;
		for (int i = 0; i < response.size(); i++)
			rss += Math.pow(response.get(i) - samples.get(i)[ta], 2);
		return rss;
	}
}
