package spawnn.utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class ColorUtils {
		
	public enum ColorClass {Equal, Quantile, kMeans};
	
	@Deprecated
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm ) {
		return getColorMap(valueMap, cm,  new HashSet<Double>(valueMap.values()).size(), ColorClass.Equal, true );
	}
	
	@Deprecated
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm, boolean quantile ) {
		return getColorMap(valueMap, cm,  new HashSet<Double>(valueMap.values()).size(), quantile ? ColorClass.Quantile : ColorClass.Equal, true );
	}
	
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm, ColorClass cc ) {
		return getColorMap(valueMap, cm,  new HashSet<Double>(valueMap.values()).size(), cc, true );
	}
		
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm, int nrColors, ColorClass cc, boolean allowInterpolate ) {
		Color[] cols = cm.getColorPalette( nrColors, allowInterpolate );
		Map<T, Color> colMap = new HashMap<T, Color>();
		
		List<Double> values = new ArrayList<Double>(valueMap.values());
		Collections.sort(values);
		
		if( cc == ColorClass.Quantile ) {
			int qSize = (int)Math.round((double)values.size()/cols.length);
			int curCol = 0;
			for( double v : values ) {
				
				for( Entry<T,Double> e : valueMap.entrySet() ) 
					if( e.getValue() <= v && !colMap.containsKey(e.getKey() ) ) {
						colMap.put(e.getKey(),cols[curCol]);
						break;
					}
				if( colMap.size() % qSize == 0 && curCol < cols.length - 1) // bucket full?
					curCol++;
			}
		} else if( cc == ColorClass.Equal ){ // eq intervall
			double min = Collections.min(values);
			double max = Collections.max(values);
			double ivSize = (max-min)/cols.length;
			
			for( Entry<T,Double> e : valueMap.entrySet() ) 
				for( int i = 0; i < cols.length; i++ ) 
					if( min + i*ivSize <= e.getValue() && e.getValue() <= min + (i+1)*ivSize + Math.pow(10, -10))
						colMap.put(e.getKey(), cols[i]);
		} else { // k-means
			System.out.println("kMeans, "+cols.length+", "+nrColors);
			List<double[]> v = new ArrayList<>();
			for( Double d : values )
				v.add( new double[]{d});
			
			Dist<double[]> dist = new EuclideanDist();
			
			Map<double[], Set<double[]>> c = null ;
			double bestQE = 0;
			
			int noImpro = 0;
			while( noImpro++ < 20 ) { // random restarts
				Map<double[], Set<double[]>> tmp = Clustering.kMeans( v, cols.length, dist );
				double qe = DataUtils.getMeanQuantizationError(tmp, dist);
				if( c == null || qe <  bestQE ) {
					bestQE = qe;
					c = tmp;
					noImpro = 0;
					System.out.println(qe);
				}
			}
						
			List<Double> centroids = new ArrayList<>();
			for( double[] d : c.keySet() )
				centroids.add(d[0]);
			Collections.sort(centroids);
			System.out.println(centroids);
			
			for( Entry<T,Double> e : valueMap.entrySet() ) {
				int best = 0;
				for( int i = 1; i < centroids.size(); i++ )
					if( Math.abs( centroids.get(i)-e.getValue() ) < Math.abs(centroids.get(best)-e.getValue() ) )
						best = i;
				colMap.put(e.getKey(),cols[best]);
			}
		}
		
		if( colMap.size() != valueMap.size() ) {
			for( T d : valueMap.keySet() )
				if( !colMap.containsKey(d) )
					System.out.println(valueMap.get(d)+" nicht drin!");
			throw new RuntimeException("Not all elements got a color! "+colMap.size()+","+valueMap.size());
		}
		return colMap;
	}
}
