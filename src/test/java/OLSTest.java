import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import chowClustering.ChowClustering;
import nnet.SupervisedUtils;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;

public class OLSTest {

	public static void main(String[] args) {
		Random r = new Random();
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 100 ) {
			double a = Math.pow(r.nextDouble(),2);
			double b = r.nextDouble();
			double c = r.nextDouble();
			samples.add( new double[]{
					a, b, c, a*b+c
			});
		}
		
		int[] fa = new int[]{0,1,2};
		int ta = 3;
		
		List<Set<double[]>> ct = new ArrayList<>( Clustering.kMeans(samples, 20, new EuclideanDist(fa)).values() );
		System.out.println(ChowClustering.minClusterSize(ct));
		
		{
		double[][] x = ChowClustering.getX(ct, samples, fa, true);
		double[] y = ChowClustering.getY(samples,ta);
		List<Double> residuals = ChowClustering.getResidualsLM(x, y, x, y);
		int nrParams = x[0].length;
		double ss = ChowClustering.getSumOfSquares(residuals);
		double aic = SupervisedUtils.getAICc(ss / samples.size(), nrParams, samples.size());
		System.out.println(nrParams+","+ss+","+aic);
		}
		
		{
		List<Double> residuals = ChowClustering.getResidualsLM(ct, samples, samples, fa, ta);
		int nrParams = ct.size() * (fa.length + 1); // + ic + error var (or so)
		double ss = ChowClustering.getSumOfSquares(residuals);
		double aic = SupervisedUtils.getAICc(ss / samples.size(), nrParams, samples.size());
		System.out.println(nrParams+","+ss+","+aic);
		}	
	}
}
