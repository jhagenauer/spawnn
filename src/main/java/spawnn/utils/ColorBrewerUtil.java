package spawnn.utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class ColorBrewerUtil {
	
	public static enum ColorMode { Gray, Temp, Reds, Blues, Greys, Spectral, Set1, Set2, Set3, Paired  };
	
	public static <T> Map<T, Color> valuesToColors(Map<T, Double> valueMap, ColorMode cm ) {
		Map<T, Color> colors = new HashMap<T, Color>();
		
		List<Double> l = new ArrayList<Double>(new HashSet<Double>(valueMap.values()));
		Collections.sort(l);
		double min = l.get(0);
		double max = l.get(l.size()-1);
		Color[] cols = new Color[l.size()];
		
		// assign colors to values
		if( cm == ColorMode.Temp) {
			for( int i = 0; i < l.size(); i++ ) {
				float v = (float)((i*(max-min)/cols.length)/(max-min));
				cols[i] = Drawer.getColor(v);
			}
		} else if( cm == ColorMode.Gray ) {
			for( int i = 0; i < l.size(); i++ ) {
				float v = (float)((i*(max-min)/cols.length)/(max-min));
				cols[i] = new Color(1 - v, 1 - v, 1 - v);
			}
		} else if( cm == ColorMode.Reds ) {
			cols = ColorBrewer.Reds.getColorPalette(l.size());
		}  else if( cm == ColorMode.Set1 ) {
			cols = ColorBrewer.Set1.getColorPalette(l.size());
		}  else if( cm == ColorMode.Set2 ) {
			cols = ColorBrewer.Set2.getColorPalette(l.size());
		} else if( cm == ColorMode.Set3 ) {
			cols = ColorBrewer.Set3.getColorPalette(l.size());
		} else if( cm == ColorMode.Paired ) {
			cols = ColorBrewer.Paired.getColorPalette(l.size());
		} else if( cm == ColorMode.Spectral ) {
			cols = ColorBrewer.Spectral.getColorPalette(l.size());
		} else if( cm == ColorMode.Greys ) {
			cols = ColorBrewer.Greys.getColorPalette(l.size());
		} else if( cm == ColorMode.Blues ) {
			cols = ColorBrewer.Blues.getColorPalette(l.size());
		} 
		
		for (T p : valueMap.keySet()) {
			int idx = l.indexOf(valueMap.get(p));
			colors.put(p, cols[idx]);
		}
		return colors;
	}
}
