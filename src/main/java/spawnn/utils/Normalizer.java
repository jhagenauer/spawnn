package spawnn.utils;

import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class Normalizer {
	
	public enum Transform {
		pow2, log, sqrt, div, zScore, scale01, pca
	}

	int[] fa;
	Transform t;
	SummaryStatistics[] ds;
	
	public Normalizer( Transform t, List<double[]> samples ) {
		this(t,samples,null);
	}
	
	public Normalizer( Transform t, List<double[]> samples, int[] faa ) {
		if( faa == null ) {
			this.fa = new int[samples.get(0).length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = faa;
		this.t = t;
		
		ds = new SummaryStatistics[fa.length];
		for (int i = 0; i < fa.length; i++)
			ds[i] = new SummaryStatistics();
		for (double[] d : samples)
			for (int i = 0; i < fa.length; i++)
				ds[i].addValue(d[fa[i]]);
		
		normalize(samples);
	}
	
	public void normalize(List<double[]> samples) {
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			for (int j = 0; j < fa.length; j++) {
				if (t == Transform.zScore)
					d[fa[j]] = (d[fa[j]] - ds[j].getMean()) / ds[j].getStandardDeviation();
				else if (t == Transform.scale01)
					d[fa[j]] = (d[fa[j]] - ds[j].getMin()) / (ds[j].getMax() - ds[j].getMin());
				else
					throw new RuntimeException(t+" not supported!");
			}
		}
	}

	@Deprecated
	public static void transform(List<double[]> samples, Transform t) {
		int[] fa = new int[samples.get(0).length];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i;
		transform(samples, fa, t);
	}

	@Deprecated
	public static void transform(List<double[]> samples, int[] fa, Transform t) {
		SummaryStatistics[] ds = new SummaryStatistics[fa.length];
		for (int i = 0; i < fa.length; i++)
			ds[i] = new SummaryStatistics();
		for (double[] d : samples)
			for (int i = 0; i < fa.length; i++)
				ds[i].addValue(d[fa[i]]);
	
		RealMatrix v = null;
		if (t == Transform.pca) {
			RealMatrix matrix = new Array2DRowRealMatrix(samples.size(), fa.length);
			for (int i = 0; i < samples.size(); i++)
				for (int j = 0; j < fa.length; j++)
					matrix.setEntry(i, j, samples.get(i)[fa[j]]);
	
			SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
			v = svd.getU().multiply(svd.getS());
		}
	
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			for (int j = 0; j < fa.length; j++) {
				if (t == Transform.log)
					d[fa[j]] = Math.log(d[fa[j]]);
				else if (t == Transform.pow2)
					d[fa[j]] = Math.pow(d[fa[j]], 2);
				else if (t == Transform.sqrt)
					d[fa[j]] = Math.sqrt(d[fa[j]]);
				else if (t == Transform.div)
					d[fa[j]] = 1.0 / d[fa[j]];
				else if (t == Transform.zScore)
					d[fa[j]] = (d[fa[j]] - ds[j].getMean()) / ds[j].getStandardDeviation();
				else if (t == Transform.scale01)
					d[fa[j]] = (d[fa[j]] - ds[j].getMin()) / (ds[j].getMax() - ds[j].getMin());
				else if (t == Transform.pca)
					d[fa[j]] = v.getEntry(i, j);
			}
		}
	}

	@Deprecated
	public static void transform(List<double[]> samples, int fa, Transform t) {
		Normalizer.transform(samples, new int[]{fa}, t);
	}
}
