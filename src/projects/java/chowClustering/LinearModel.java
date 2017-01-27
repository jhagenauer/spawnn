package chowClustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import nnet.SupervisedUtils;

public class LinearModel {
	private List<DoubleMatrix> betas;
	private List<Double> residuals = null;
	List<double[]> samples;
	int[] fa;
	int ta, maxIter;
	double lambda; 
	private double rss = -1;
	
	boolean zScore;
	List<Set<double[]>> cluster;
	private List<double[]> means = new ArrayList<>(), sds = new ArrayList<>();
	
	public LinearModel(List<double[]> samples, int[] fa, int ta, boolean zScore ) {
		this( samples, null, fa, ta, zScore, -1.0);
	}
	
	public LinearModel(List<double[]> samples, int[] fa, int ta, boolean zScore, double lambda ) {
		this( samples, null, fa, ta, zScore, -1.0);
	}
		
	public LinearModel(List<double[]> samples, List<Set<double[]>> cluster,int[] fa, int ta, boolean zScore ) {
		this( samples, cluster, fa, ta, zScore, -1.0);
	}
		
	public LinearModel(List<double[]> samples, List<Set<double[]>> cluster, int[] fa, int ta, boolean zScore, double lambda ) {
		this.samples = samples;
		this.fa = fa;
		this.ta = ta;
		this.zScore = zScore;
		this.lambda = lambda;
		this.betas = new ArrayList<>();
		
		if( cluster == null ) {
			this.cluster = new ArrayList<>();
			this.cluster.add( new HashSet<>(samples));
		} else {
			this.cluster = cluster;
		}
		
		if( lambda > 0 && !zScore )
			System.out.println("Warning: Ridge regression without zScore "+lambda);
				
		for( int j = 0; j < this.cluster.size(); j++ ) {
			Set<double[]> c = this.cluster.get(j);
			
			List<double[]> l = new ArrayList<double[]>(c);
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
					
			DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( l, ta) );
			DoubleMatrix Xt = X.transpose();
			DoubleMatrix XtX = Xt.mmul(X);					
			
			if( lambda > 0 ) { // ridge regression					
				XtX.addi( DoubleMatrix.eye(XtX.columns).muli(lambda) );							
			}
			betas.add(Solve.solve(XtX, Xt.mmul(Y))); 
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
	
	public double getR2(List<double[]> samples) {
		return SupervisedUtils.getR2(getPredictions(samples, fa), samples, ta);
	}
	
	public double getRSS() {
		if( rss < 0 ) {
			rss = 0;
			for( double d : getResiduals() )
				rss += d*d;
		}
		return rss;
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
	
	public List<Set<double[]>> getCluster() {
		return cluster;
	}
}