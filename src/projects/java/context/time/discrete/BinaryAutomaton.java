package context.time.discrete;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import cern.colt.bitvector.BitVector;

public class BinaryAutomaton {
	
	private static Logger log = Logger.getLogger(BinaryAutomaton.class);
	
	public static void main(String[] args) {
				
		int length = 40000;
		Random r = new Random();
				
		List<double[]> samples = new ArrayList<double[]>();
		int state = r.nextDouble() < 3.0/7.0 ? 1 : 0;
		for( int i = 0; i < length; i++ ) {
			
			if( state == 1 && r.nextDouble() < 0.4 ) 
				state = 0;
			else if( state == 0 && r.nextDouble() < 0.3 )
				state = 1;
									
			double[] s = new double[]{i, state};
			samples.add( s );
		}
		
		//DataUtils.writeCSV("data/somsd/binary.csv", samples, new String[]{"time","value"} );
		//log.debug("csv written.");
		//System.exit(1);
		
		BitVector bv = new BitVector(samples.size());
		for( int i = 0; i < samples.size(); i++ )
			bv.put(i, (int)samples.get(i)[1] == 1 );
				
		// get all sequences up to length lim
		final Map<BitVector,Integer> contextMap = getContext(bv, 25);
		log.debug("contextMap.size: "+contextMap.size());
		
		List<BitVector> sorted = new ArrayList<BitVector>(contextMap.keySet()); //TODO maybe better results with
		Collections.sort(sorted, new Comparator<BitVector>() {
			@Override
			public int compare(BitVector o1, BitVector o2) {
				return Double.compare( contextMap.get(o1), contextMap.get(o2) );
			}
		});
		Collections.reverse(sorted);
										
		// get n most frequent sequences
		List<BitVector> top = sorted.subList(0, 150 );
		for( BitVector b : top ) {
					
			StringBuffer sb = new StringBuffer();
			for( int i = 0; i < b.size(); i++ )
				if( b.get(i) )
					sb.append("1");
				else
					sb.append("0");
			
			log.info(sb+"\t\t"+(int)contextMap.get(b) );
		}
						
		Map<BitVector,Integer> actualCount = new HashMap<BitVector,Integer>();
		for( int i = 1; i < bv.size(); i++) {
			
			// get longest match from top
			BitVector bm = null;
			for( BitVector t : top ) {
				
				if( i - t.size()  < 0 )
					continue;
				
				BitVector part = bv.partFromTo(i-t.size(), i-1);
				if( part.equals(t) && ( bm == null || bm.size() < part.size() ) ) 
					bm = t;
			}
			// log.debug(toString(bm)+" is longest match for "+toString(bv.partFromTo(0, i-1))+": "+i);
			
			if( !actualCount.containsKey(bm) )
				actualCount.put(bm, 0);
			actualCount.put(bm, actualCount.get(bm)+1);
		}
		log.debug("size: "+actualCount.size());
		
		// find optimal optimizer ;)
		double totNumContext = 0;
		for( BitVector b: actualCount.keySet() )
			totNumContext += actualCount.get(b);
		
		double sum = 0;
		for( BitVector b : actualCount.keySet() ) {
			
			// prob of occurence		
			double pi = (double)actualCount.get(b)/totNumContext; 

			sum += pi * b.size();
		}
		log.debug("depth of quantizier: "+sum+", "+totNumContext);
					
		List<String> nodes = new ArrayList<String>( getNodes( new HashSet<BitVector>(top), 0, new double[]{ 0, 0 }, new BitVector(0), 10 ) );
			
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter("output/tree.dat"));
			for( String s : nodes )
				bw.write(s);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try { bw.close(); } catch( Exception e ) {}
		}
		
	}
	
	public static List<String> getNodes( Set<BitVector> top, int level, double[] curPos, BitVector bv, int maxLevel ) {
		List<String> nodes = new ArrayList<String>();
		
		if( level > maxLevel ) 
			return nodes;
		
		BitVector nbv = new BitVector(bv.size()+1);
		for( int i = 0; i < bv.size(); i++ )
			nbv.put(i,bv.get(i));
				
		nbv.put( bv.size(), true );
		double[] newPosR = new double[]{ curPos[0] + 1.0/Math.pow(2, level), level+1 }; 
		List<String> lr = getNodes( top, level + 1, newPosR, nbv, maxLevel );
		if( !lr.isEmpty() || top.contains(nbv) ) {
			nodes.addAll(lr);
			nodes.add( curPos[0]+" "+curPos[1]+"\n"+newPosR[0]+" "+newPosR[1]+"\n\n" );
		}
			
		nbv.put( bv.size(), false ); 
		double[] newPosL = new double[]{curPos[0] - 1.0/Math.pow(2, level), level+1 }; 
		List<String> ll = getNodes( top, level+1, newPosL, nbv, maxLevel );
		if( !ll.isEmpty() || top.contains(nbv) ) {
			nodes.addAll(ll);
			nodes.add( curPos[0]+" "+curPos[1]+"\n"+newPosL[0]+" "+newPosL[1]+"\n\n" );
		}
										
		return nodes;
	}
	
	public static String toString(BitVector bv ) {
		StringBuffer sb = new StringBuffer();
		for( int i = 0; i < bv.size(); i++ )
			if( bv.get(i) )
				sb.append("1");
			else
				sb.append("0");
		return sb.toString();
	}
	
	// test
	public static Map<BitVector,Integer> getContext( BitVector bv, int lim ) {
		final Map<BitVector,Integer> countMap = new HashMap<BitVector,Integer>();
		for( int i = 0; i < bv.size(); i++ ) {
			for( int j = i; j < bv.size(); j++ ) {
				
				if( j-i > lim )
					continue;
							
				BitVector v = bv.partFromTo(i, j);							
				if( !countMap.containsKey(v) )
					countMap.put(v, 1);
				else
					countMap.put(v, countMap.get(v)+1);

			}
		}
		return countMap;
	}

}
