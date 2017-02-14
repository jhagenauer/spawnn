package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Solve;

import nnet.SupervisedUtils;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class LinearModel {
	private List<DoubleMatrix> betas;
	private List<Double> residuals = null;
	List<double[]> samples;
	int[] fa;
	int ta, maxIter;
	private double rss = -1;
	
	boolean zScore;
	List<Set<double[]>> cluster;
	private List<double[]> means = new ArrayList<>(), sds = new ArrayList<>();
	
	public LinearModel(List<double[]> samples, int[] fa, int ta, boolean zScore ) {
		this( samples, null, fa, ta, zScore);
	}
	
	public LinearModel(List<double[]> samples, List<Set<double[]>> cluster,int[] fa, int ta, boolean zScore ) {
		this.samples = samples;
		this.fa = fa;
		this.ta = ta;
		this.zScore = zScore;
		this.betas = new ArrayList<>();
		
		if( cluster == null ) {
			this.cluster = new ArrayList<>();
			this.cluster.add( new HashSet<>(samples));
		} else {
			this.cluster = cluster;
		}
				
		for( int j = 0; j < this.cluster.size(); j++ ) {
			Set<double[]> c = this.cluster.get(j);
			
			List<double[]> l = new ArrayList<double[]>(c);
			
			DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( l, ta) );
			DoubleMatrix X;
			if( zScore ) {				
				double[] mean = new double[fa.length], sd = new double[fa.length];
				for( int i = 0; i < fa.length; i++ ) {
					SummaryStatistics ss = new SummaryStatistics();
					for( double[] d : c )
						ss.addValue( d[fa[i]] );				
					mean[i] = ss.getMean();
					sd[i] = ss.getStandardDeviation();
				}
				means.add(mean);
				sds.add(sd);					
				X = new DoubleMatrix( LinearModel.getX( l, fa, mean, sd, true) );	
			} else 
				X = new DoubleMatrix( LinearModel.getX( l, fa, true) );	
			
			// calculate beta
			DoubleMatrix Xt = X.transpose();
			DoubleMatrix XtX = Xt.mmul(X);
			DoubleMatrix beta = Solve.solve(XtX, Xt.mmul(Y));
			betas.add(beta);
						
			// calculate std beta error, not here because expsensive and rarely needed
			/*DoubleMatrix betaVar = Solve.pinv(XtX);			
			DoubleMatrix residuals = Y.sub( X.mmul(beta) ); // predictions
			
			double sigma = residuals.dot(residuals)/(X.rows-X.columns);
			DoubleMatrix stdBetaError = MatrixFunctions.sqrt( betaVar.diag().mul(sigma) );*/
		}			
	}
				
	public List<Double> getResiduals() {
		if( residuals == null ) {
			List<Double> predictions = getPredictions(samples, fa);
			residuals = new ArrayList<Double>();
			for( int i = 0; i < samples.size(); i++ )
				residuals.add( samples.get(i)[ta] - predictions.get(i) );
		} 
		return residuals;	
	}
		
	public double getRSS() {
		if( rss < 0 ) {
			rss = 0;
			for( double d : getResiduals() )
				rss += d*d;
		}
		return rss;
	}
	
	public double getRSS( int i ) {
		List<double[]> l = new ArrayList<>(cluster.get(i));
		List<Double> pred = getPredictions(l, fa);
		return SupervisedUtils.getResidualSumOfSquares(pred, l, ta);
	}
				
	public List<Double> getPredictions( List<double[]> samples, int[] faPred ) {		
		Double[] predictions = new Double[samples.size()];
							
		for (int l = 0; l < betas.size(); l++ ) {
			Set<double[]> c = cluster.get(l);
			
			List<double[]> subSamples = new ArrayList<>();
			Map<Integer,Integer> idxMap = new HashMap<Integer,Integer>(); 
			for( int i = 0; i < samples.size(); i++ ) {
				double[] d = samples.get(i);
				if( cluster.size() == 1 || c.contains( d ) ) {
					idxMap.put(subSamples.size(), i);
					subSamples.add(d);
				}
			}
			
			if( subSamples.isEmpty() )
				continue;
							
			DoubleMatrix X;
			if( zScore )
				X = new DoubleMatrix( LinearModel.getX( subSamples, faPred, means.get(l), sds.get(l), true) );
			else
				X = new DoubleMatrix( LinearModel.getX( subSamples, faPred, true) );
			
			DoubleMatrix beta = betas.get(l);
			double[] p = X.mmul(beta).data;
			for( int i = 0; i < p.length; i++ )
				predictions[idxMap.get(i)] = p[i];				
		}
		return Arrays.asList(predictions);
	}
	
	public static double[] getY(List<double[]> samples, int ta) {
		double[] y = new double[samples.size()];
		for (int i = 0; i < samples.size(); i++)
			y[i] = samples.get(i)[ta];
		return y;
	}
	
	public static double[][] getX(List<double[]> samples, int[] fa, boolean addIntercept) {
		return getX(samples, fa, null, null, addIntercept);
	}
	
	public static double[][] getX(List<double[]> samples, int[] fa, double mean[], double sd[], boolean addIntercept) {		
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			
			x[i] = new double[fa.length + (addIntercept ? 1 : 0) ];
			x[i][x[i].length - 1] = 1.0; // gets overwritten if !addIntercept
			for (int j = 0; j < fa.length; j++) {
				if( mean != null && sd != null )
					x[i][j] = (d[fa[j]]-mean[j])/sd[j];
				else
					x[i][j] = d[fa[j]];
			}
		}
		return x;
	}
	
	public double[] getBeta(int i) {
		return betas.get(i).data;
	}
	
	public double[] getBetaStdError(int i) {
		List<double[]> l = new ArrayList<>(cluster.get(i));
		DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( l, ta) );
		DoubleMatrix X;
		if( zScore )
			X = new DoubleMatrix( LinearModel.getX( l, fa, means.get(i), sds.get(i), true) );
		else
			X = new DoubleMatrix( LinearModel.getX( l, fa, true) );
		DoubleMatrix XtX = X.transpose().mmul(X);
		
		DoubleMatrix betaVar = Solve.pinv(XtX);			
		DoubleMatrix residuals = Y.sub( X.mmul(betas.get(i)) ); // predictions
					
		double sigma = residuals.dot(residuals)/(X.rows-X.columns);
		DoubleMatrix stdBetaError = MatrixFunctions.sqrt( betaVar.diag().mul(sigma) );
		return stdBetaError.data;
	}
		
	public List<Set<double[]>> getCluster() {
		return cluster;
	}
	
	public static void main(String[] args) {
		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/mtcars.csv"), new int[]{}, true);
		for( int i = 0; i < df.names.size(); i++ )
			System.out.println(i+": "+df.names.get(i));
		
		LinearModel lm = new LinearModel(df.samples, new int[]{3,5,9}, 0, false);
		double[] beta = lm.getBeta(0);
		double[] stdError = lm.getBetaStdError(0);
		
		TDistribution td = new TDistribution(df.samples.size()-beta.length); 
		for( int i = 0; i < beta.length; i++ ) {
			double tValue = beta[i]/stdError[i];
			double pValue = 2*(td.cumulativeProbability(-Math.abs(tValue) ) );
			System.out.println(beta[i]+" "+tValue+" "+pValue);
		}
	}
}