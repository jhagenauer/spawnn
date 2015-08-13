package regionalization.tabu;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import regionalization.ga.WCSSCostCalulator;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;

public class RegioTabuSearch {

	private static Logger log = Logger.getLogger(RegioTabuSearch.class);
	
	Random r = new Random();
	
	public int f = 0;
	public int tlLength = 10;
	public int penaltyDur = 25;
	public int penaltyStart = 50;
	
	public TabuIndividual search( TabuIndividual init ) {

		TabuList<TabuMove> tl = new TabuList<TabuMove>(); // recency
		Map<Integer,Integer> fr = new HashMap<Integer,Integer>(); // frequency
				
		TabuIndividual curBest = init; // aktuell
		
		TabuIndividual globalBest = curBest; // global Best
		int noImpro = 0;
			
		int k = 0;
		while( noImpro < 300 ) {
			// get best non-tabu or global best move
			TabuIndividual localBest = null; // best local non-tabu
			TabuMove localBestMove = null;
						
			List<TabuMove> moves = curBest.getAllMoves();	
						
			for( TabuMove curMove : moves ) { // get best move for curBest
				
				double penalty = 1;
				if( noImpro >= penaltyStart && noImpro % penaltyStart < penaltyDur ) {
					int key = ((RegioTabuMove)curMove).getFrom();	
					if( fr.containsKey(key) )
						penalty = (double)fr.get(key)/Collections.max(fr.values()); 
				}
								
				TabuIndividual cur = curBest.applyMove( curMove );			
				if( ( !tl.isTabu( curMove ) && ( localBest == null || cur.getValue() + f * penalty < localBest.getValue() ) )
						|| cur.getValue() < globalBest.getValue() ) { // Aspiration by objecive
							localBest = cur;
							localBestMove = curMove;									
				}
			}
																							
			if( localBest == null ) //Aspiration by default, TODO: must be implemented
				log.error("Only tabu-moves possible!");

			//log.debug(k+", localbest: "+localBest.getValue() );
											
			if( localBest.getValue() < globalBest.getValue() ) { 
				log.debug("Found new global best: "+globalBest.getValue()+", k: "+k );	
				globalBest = localBest;
				noImpro = 0;
				fr.clear();
			} else
				noImpro++;
			
			// update frequency list, only saving keys is better than saving the whole move (sagt wer?)
			int key = ((RegioTabuMove)localBestMove).getFrom();
			if( !fr.containsKey(key) )
				fr.put(key, 1);
			else
				fr.put(key, fr.get(key)+1);
			
			curBest = localBest; // update curBest
			TabuMove invMove = localBestMove; 
			if( !tl.isTabu( invMove ) ) 
				tl.add( invMove, tlLength ); // tenure
						
			tl.step();
			k++;
		}
		return globalBest;
	}

	public static void main(String[] args) {
		int numRegions = 7;
		
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/regionalization/100rand.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/100rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/200rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/500rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/500rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/1000rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/1000rand.shp"), new int[] {}, true);
		//final List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		//final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/lisbon/lisbon.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/lisbon/lisbon.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };
		//int[] fa = new int[] { 1 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);
		
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/100rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/500rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/1000rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/lisbon/lisbon_queen.ctg");
				
		RegioTabuIndividual init = new RegioTabuIndividual( new ArrayList<double[]>(samples), numRegions, new WCSSCostCalulator(new EuclideanDist(fa)), cm);
		
		RegioTabuSearch ts = new RegioTabuSearch();
		RegioTabuIndividual result = (RegioTabuIndividual)ts.search( init );
									
		log.debug("Heterogenity: "+result.getValue() );
						
		try { 
			Drawer.geoDrawCluster( result.getCluster(), samples, geoms, new FileOutputStream("output/ts.png"), true ); 
		} catch(FileNotFoundException e) {
			e.printStackTrace(); 
		}	
	}
}
