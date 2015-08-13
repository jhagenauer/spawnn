package regionalization.zd;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import regionalization.ga.InequalityCalculator;
import regionalization.tabu.RegioTabuIndividual;
import regionalization.tabu.RegioTabuSearch;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;

public class ZDTabuSearch {

	private static Logger log = Logger.getLogger(ZDTabuSearch.class);
		
	public static void main(String[] args) {
		int numRegions = 7;
				
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/lisbon/lisbon.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/lisbon/lisbon.shp"), new int[] {}, true);
		int[] fa = new int[] { 1 };
		Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/lisbon/lisbon_queen.ctg");
				
		double mean = 0;
		for( double[] d : samples )
			mean += d[fa[0]]/numRegions;
		
		RegioTabuIndividual init = new RegioTabuIndividual( new ArrayList<double[]>(samples), numRegions, new InequalityCalculator(fa, mean), cm );
									
		RegioTabuSearch ts = new RegioTabuSearch();
		RegioTabuIndividual result = (RegioTabuIndividual)ts.search( init );
		
		log.debug("Global best: "+result.getValue());
		
		try { 
			Drawer.geoDrawCluster( result.getCluster(), samples, geoms, new FileOutputStream("output/lisbon_tabu.png"), true ); 
		} catch(FileNotFoundException e) {
			e.printStackTrace(); 
		}	
	}
}
