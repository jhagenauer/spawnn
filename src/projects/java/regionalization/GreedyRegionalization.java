package regionalization;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

public class GreedyRegionalization {
	
	private static Logger log = Logger.getLogger(GreedyRegionalization.class);
			
	public static void main(String[] args) {
		Random r = new Random();
		
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/200rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/regionalization/500rand.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/500rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);

		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/500rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
				
		List<Set<double[]>> map = new ArrayList<Set<double[]>>();
		Set<double[]> init = new HashSet<double[]>();
		while( init.size() != 15 ) {
			double[] x = samples.get(r.nextInt(samples.size()));
			if( !init.contains(x) )
				init.add(x);
		}
		
		for( double[] d : init ) {
			Set<double[]> s = new HashSet<double[]>();
			s.add(d);
			map.add( s );
		}
								
		// check if all sampels are mapped
		List<double[]> l = new ArrayList<double[]>();
		for( Set<double[]> s : map ) {
			l.addAll( s );
		}
			
		List<double[]> diff = new ArrayList<double[]>(samples);
		diff.removeAll(l);
		if( !diff.isEmpty() )
			log.warn("not all samples are mapped! ("+diff.size()+" of "+samples.size()+")");
								
		// fix mapping greedy
		List<double[]> toAdd = new ArrayList<double[]>(diff);
		while( !toAdd.isEmpty() ) {
			List<double[]> redo = new ArrayList<double[]>();
								
			for( double[] d : toAdd ) {
					
				Set<double[]> best = null;
				double bestHeterogenity = Double.MAX_VALUE;
					
				for( Set<double[]> s : map ) {
					s.add(d);
					if( RegionUtils.isContiugous(cm, s) && ( best == null || RegionUtils.getHeterogenity(map, fa) < bestHeterogenity ) ) {
						best = s;
						bestHeterogenity = RegionUtils.getHeterogenity(map, fa);
					}
					s.remove(d);
				}
				if( best != null ) 
					best.add(d);
				else
					redo.add(d); 
			}
			toAdd = redo;
		}
													
		List<Set<double[]>> l2 = new ArrayList<Set<double[]>>();
		for( Set<double[]> s : map ) 
			l2.add( new HashSet<double[]>(s) );	
													
		int sum = 0;
		for( Set<double[]> s : map )
			sum += s.size();
				
		log.debug("mapped: "+sum);		
		log.debug("Heterogenity map: "+RegionUtils.getHeterogenity(map, fa));
		
		try {
			Drawer.geoDrawCluster(map, samples, geoms, new FileOutputStream("output/greedy.png"), true);
		} catch (FileNotFoundException e) {
			
		}
					
	}
}
