import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import chowClustering.ChowClustering;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;

public class OLSTest {

	public static void main(String[] args) {
		Random r = new Random(0L);
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 100 ) {
			double a = Math.pow(r.nextDouble(),2);
			double b = r.nextDouble();
			samples.add( new double[]{
					a, b, a*b
			});
		}
		
		int[] fa = new int[]{0,1};
		int ta = 2;
		
		List<Set<double[]>> ct = new ArrayList<>( Clustering.kMeans(samples, 2, new EuclideanDist(fa)).values() ) ;
		{
			double[][] x = ChowClustering.getX(ct, samples, fa );	
			double[] y = ChowClustering.getY(samples,ta);
			List<Double> residuals = null;
			residuals = ChowClustering.getResidualsLM(x, y, x, y);
					
			int nrParams = x[0].length;
			double ss = ChowClustering.getSumOfSquares(residuals);
			System.out.println(nrParams+","+ss);
		}
		
		Collections.shuffle(ct);
		{
			double[][] x = ChowClustering.getX(ct, samples, fa );	
			double[] y = ChowClustering.getY(samples,ta);
			List<Double> residuals = null;
			residuals = ChowClustering.getResidualsLM(x, y, x, y);
						
			int nrParams = x[0].length;
			double ss = ChowClustering.getSumOfSquares(residuals);
			System.out.println(nrParams+","+ss);
		}
		/*
		
		{
			double[][] x = ChowClustering.getX(ct, samples, fa, true );	
			double[] y = ChowClustering.getY(samples,ta);
			List<Double> residuals = null;
			residuals = ChowClustering.getResidualsLM(x, y, x, y);
					
			int nrParams = x[0].length;
			double ss = ChowClustering.getSumOfSquares(residuals);
			//double aic = SupervisedUtils.getAICc(ss / samples.size(), nrParams, samples.size());
			System.out.println(nrParams+","+ss);
		}
		
		{
		List<Double> residuals = ChowClustering.getResidualsLM(ct, samples, samples, fa, ta);
		int nrParams = ct.size() * (fa.length + 1); // + ic + error var (or so)
		double ss = ChowClustering.getSumOfSquares(residuals);
		//double aic = SupervisedUtils.getAICc(ss / samples.size(), nrParams, samples.size());
		System.out.println(nrParams+","+ss);
		}	
		
		*/
	}
}
