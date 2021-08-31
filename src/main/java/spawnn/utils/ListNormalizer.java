package spawnn.utils;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class ListNormalizer extends Normalizer {
	
	int[] fa;
	Transform[] tt;
	SummaryStatistics[] ds;
	
	public ListNormalizer( Transform t, List<double[]> samples ) {
		this( new Transform[] {t},samples,null);
	}
	
	public ListNormalizer( Transform[] t, List<double[]> samples, int[] faa ) {
		if( faa == null ) {
			this.fa = new int[samples.get(0).length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = faa;
				
		ds = new SummaryStatistics[fa.length];
		for (int i = 0; i < fa.length; i++)
			ds[i] = new SummaryStatistics();
		for (double[] d : samples)
			for (int i = 0; i < fa.length; i++) {
				double v = d[fa[i]];
				if( !Double.isNaN(v) )
					ds[i].addValue(v);
			}
		normalize(samples);
		
		this.tt = t;
	}
	
	@Deprecated
	public ListNormalizer( Transform t, List<double[]> samples, int[] faa ) {
		this( new Transform[] {t},samples,faa);
	}
	
	public void normalize(List<double[]> samples) {
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			for (int j = 0; j < fa.length; j++) 
				for( Transform t : tt ){
					if (t == Transform.zScore) 
						d[fa[j]] = (d[fa[j]] - ds[j].getMean()) / ds[j].getStandardDeviation();
					else if (t == Transform.scale01)
						d[fa[j]] = (d[fa[j]] - ds[j].getMin()) / (ds[j].getMax() - ds[j].getMin());
					else if( t == Transform.sqrt )
						d[fa[j]] = Math.sqrt(d[fa[j]]);
					else if( t == Transform.log )
						d[fa[j]] = Math.log(d[fa[j]]);
					else if( t == Transform.log1 )
						d[fa[j]] = Math.log( d[fa[j]]+1.0 );
					else if( t == Transform.none )
						d[fa[j]] = d[fa[j]];
					
					else
						throw new RuntimeException(t+" not supported!");
				}
		}
	}
		
	public void denormalize(double[] d, int j ) {
		for (int i = 0; i < d.length; i++) 
			for( Transform t : tt ){
				if (t == Transform.zScore) 
					d[i] = ds[j].getStandardDeviation() * d[i] + ds[j].getMean();
				else if (t == Transform.scale01) 
					d[i] = (ds[j].getMax() - ds[j].getMin()) * d[i] + ds[j].getMin();
				else if( t == Transform.sqrt )
					if( d[i] < 0 )
						d[i] = Double.NaN;
					else
						d[i] = Math.pow(d[i],2);
				else if( t == Transform.log )
					d[i] = Math.exp(d[i]); // correct?
				else
					throw new RuntimeException(t+" not supported!");
			}		
	}
}
