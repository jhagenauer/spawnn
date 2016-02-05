package spawnn.utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class ColorUtils {
		
	public static ColorBrewer[] colors = new ColorBrewer[]{ 
		ColorBrewer.Blues, ColorBrewer.Reds, ColorBrewer.Greys, ColorBrewer.RdBu, 
		ColorBrewer.Spectral, ColorBrewer.Set1, ColorBrewer.Set2, ColorBrewer.Set3, ColorBrewer.Paired 
		};
		
	public static enum ColorMode { Blues, Reds, Greys, RdBu, Spectral, Set1, Set2, Set3, Paired };
	
	/* TODO FIXME The dynamic number of colors and quantile-mode are NECESSARY 
	 * so that cluster visualization of the GUI works correctly.
	 * This needs to solved in a better way.
	 * 
	 * quantil = each color same number/size, bad for clustering
	 * interva = linear color spread
	 */
	@Deprecated
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorMode cm ) {
		return getColorMap(valueMap, cm,  new HashSet<Double>(valueMap.values()).size(), false);
	}
	
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorMode cm, boolean quantile ) {
		return getColorMap(valueMap, cm,  new HashSet<Double>(valueMap.values()).size(), quantile);
	}
	
	public static <T> Map<T, Color> getColorMap( List<T> elems, List<Double> values, ColorMode cm, int nrColors, boolean quantil ) {
		Map<T,Double> vMap = new HashMap<T,Double>();
		for( int i = 0; i < elems.size(); i++ )
			vMap.put(elems.get(i), values.get(i));
		return getColorMap(vMap, cm, nrColors, quantil);
	}
		
	public static <T> Map<T, Color> getColorMap(Map<T, Double> valueMap, ColorMode cm, int nrColors, boolean quantil ) {
		Color[] cols = null;
		if( cm == ColorMode.Reds ) {
			cols = ColorBrewer.Reds.getColorPalette(nrColors);
		}  else if( cm == ColorMode.Set1 ) {
			cols = ColorBrewer.Set1.getColorPalette(nrColors);
		}  else if( cm == ColorMode.Set2 ) {
			cols = ColorBrewer.Set2.getColorPalette(nrColors);
		} else if( cm == ColorMode.Set3 ) {
			cols = ColorBrewer.Set3.getColorPalette(nrColors);
		} else if( cm == ColorMode.Paired ) {
			cols = ColorBrewer.Paired.getColorPalette(nrColors);
		} else if( cm == ColorMode.Spectral ) {
			cols = ColorBrewer.Spectral.getColorPalette(nrColors);
		} else if( cm == ColorMode.Greys ) {
			cols = ColorBrewer.Greys.getColorPalette(nrColors);
		} else if( cm == ColorMode.Blues ) {
			cols = ColorBrewer.Blues.getColorPalette(nrColors);
		} else if( cm == ColorMode.RdBu ) {
			cols = ColorBrewer.RdBu.getColorPalette(nrColors);
		} else {
			throw new RuntimeException("Unknown ColorMode");
		}
		
		Map<T, Color> colMap = new HashMap<T, Color>();
		
		List<Double> values = new ArrayList<Double>(valueMap.values());
		Collections.sort(values);
		
		if( quantil ) {
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
		} else { // eq intervall
			double min = Collections.min(values);
			double max = Collections.max(values);
			double ivSize = (max-min)/cols.length;
			
			for( Entry<T,Double> e : valueMap.entrySet() ) 
				for( int i = 0; i < cols.length; i++ ) 
					if( min + i*ivSize <= e.getValue() && e.getValue() <= min + (i+1)*ivSize + Math.pow(10, -10))
						colMap.put(e.getKey(), cols[i]);
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
