package chowClustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import chowClustering.ChowClustering.MyOLS;
import mltk.core.Attribute;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.NumericalAttribute;
import mltk.predictor.Learner.Task;
import mltk.predictor.glm.ElasticNetLearner;
import mltk.predictor.glm.GLM;
import nnet.SupervisedUtils;

public class PiecewiseLM {
	String method;
	List<Set<double[]>> cluster;
	List<double[]> betas;
	private List<Double> residuals = null;
	List<double[]> samples;
	int[] fa;
	int ta, maxIter;
	double l1, lambda; // l1 0: ridge, 1: lasso, (0, 1): elastic net
	int numParams = 0;
	private double rss = -1;
	
	public PiecewiseLM(List<double[]> samples, List<Set<double[]>> cluster, String method, int[] fa, int ta ) {
		this.samples = samples;
		this.cluster = cluster;
		this.method = method;
		this.fa = fa;
		this.ta = ta;
		boolean elasticNet = false;
		
		this.betas = new ArrayList<>();
		for (Set<double[]> c : cluster) {
							
			double[][] x = ChowClustering.getX( new ArrayList<double[]>(c), fa, true);
			double[] y = ChowClustering.getY( new ArrayList<double[]>(c), ta);
			double[] beta;
			
			if( !elasticNet ) {
				OLSMultipleLinearRegression ols = new MyOLS();
				ols.setNoIntercept(true);
				ols.newSampleData(y, x);
				beta = ols.estimateRegressionParameters();
			} else {
				ElasticNetLearner learner = new ElasticNetLearner();
				learner.setVerbose(false);
				learner.setTask(Task.REGRESSION);
				learner.setLambda(lambda);
				learner.setL1Ratio(l1);
				learner.setMaxNumIters(maxIter);
				
				List<Attribute> attrs = new ArrayList<>();
				for( int i = 0; i < fa.length; i++ )
					attrs.add( new NumericalAttribute("i", i));
				
				Instances is = new Instances( attrs );
				for( int i = 0; i < x.length; i++ )
					is.add( new Instance(x[i], y[i]));
				
				GLM glm = learner.build(is);
									
				double[] coefs = glm.coefficients(0);
				beta = Arrays.copyOf(coefs, coefs.length+1);
				beta[beta.length-1] = glm.intercept(0);
			}
			betas.add(beta);
		}			
		numParams = cluster.size() * (fa.length+1);
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
	
	public double getAICc() {
		return SupervisedUtils.getAICc_GWMODEL(getRSS()/samples.size(), numParams, samples.size());
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
			double[][] x = ChowClustering.getX( subSamples, faPred, true);
			double[] beta = betas.get(l);
			for (int i = 0; i < x.length; i++) {
				double p = 0;
				for (int j = 0; j < beta.length; j++)
					p += beta[j] * x[i][j];
				predictions[idxMap.get(i)] = p;						
			}
		}
		return Arrays.asList(predictions);
	}
	
	@Override
	public String toString() {
		return method + "," + cluster.size();
	}
}