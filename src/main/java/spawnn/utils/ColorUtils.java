package spawnn.utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jfree.util.Log;

import com.vividsolutions.jts.triangulate.quadedge.LastFoundQuadEdgeLocator;

import java.util.Set;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class ColorUtils {
		
	public enum ColorClass {Equal, Quantile, kMeans};
	
	@Deprecated
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm ) {
		return getColorMap(valueMap, cm,  ColorClass.Equal );
	}
	
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorBrewer cm, ColorClass cc ) {
		Color[] cols = cm.getColorPalette( new HashSet<Double>(valueMap.values()).size(), true );
		return getColorMap(valueMap, cc, cols );
	}
		
	public static <T> Map<T, Color> getColorMap( Map<T, Double> valueMap, ColorClass cc, Color[] cols ) {
		List<T> sortedKeys = new ArrayList<T>(valueMap.keySet());
		Collections.sort(sortedKeys,new Comparator<T>() {
			@Override
			public int compare(T o0, T o1) {
				return Double.compare(valueMap.get(o0), valueMap.get(o1));
			}
			
		});
		
		Map<T, Color> colMap = new HashMap<T, Color>();	
		if( cc == ColorClass.Quantile ) {
			double qSize = (double)valueMap.size()/cols.length;
			int curCol = 0;
			for( T t : sortedKeys ) {
				colMap.put(t, cols[curCol]);
				if( colMap.size() == (int)Math.round(qSize*(curCol+1)) && curCol < cols.length - 1) // bucket full?
					curCol++;
			}			
		} else if( cc == ColorClass.Equal ){ // eq intervall
			double min = Collections.min(valueMap.values());
			double max = Collections.max(valueMap.values());
			double ivSize = (max-min)/cols.length;
			
			for( Entry<T,Double> e : valueMap.entrySet() ) 
				for( int i = 0; i < cols.length; i++ ) 
					if( min + i*ivSize <= e.getValue() && e.getValue() <= min + (i+1)*ivSize + Math.pow(10, -10))
						colMap.put(e.getKey(), cols[i]);
		} else { // k-means
			List<double[]> v = new ArrayList<>();
			for( Double d : valueMap.values() )
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
			
			for( Entry<T,Double> e : valueMap.entrySet() ) {
				int best = 0;
				for( int i = 1; i < centroids.size(); i++ )
					if( Math.abs( centroids.get(i)-e.getValue() ) < Math.abs(centroids.get(best)-e.getValue() ) )
						best = i;
				colMap.put(e.getKey(),cols[best]);
			}
		}
		
		if( colMap.isEmpty() )
			throw new RuntimeException("No colors found/identified! ");
		
		if( colMap.size() != valueMap.size() ) 
			throw new RuntimeException("Not all elements got a color! "+colMap.size()+"; "+valueMap.size());
		
		return colMap;
	}
}
