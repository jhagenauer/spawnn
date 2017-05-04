package inc_llm;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/* Bidirectional connections, could be used for any growing self-organizing network.
 * Currently only used for IncLLM.
 * Beware to always manage A-B and B-A 
 */
public class ConnectionsIncLLM {
	Map<double[],Map<double[],int[]>> cons = new HashMap<>();
	//int[] oldest = null;
	
	public void add( double[] a, double[] b ) {
		int[] i = new int[]{ 0 };
		
		if( !cons.containsKey(a) )
			cons.put(a, new HashMap<double[],int[]>() );
		if( cons.get(a).containsKey(b) ) // update age
			cons.get(a).get(b)[0] = i[0];	
		else // add new 
			cons.get(a).put(b, i);
		
		if( !cons.containsKey(b) )
			cons.put(b, new HashMap<double[],int[]>() );
		if( cons.get(b).containsKey(a) ) // update age
			cons.get(b).get(a)[0] = i[0];
		else // add new 
			cons.get(b).put(a, i);
		
		//if( oldest == null || i[0] > oldest[0] ) oldest = i;
	}
	
	public int remove( double[] a, double[] b ) {
		int[] r = cons.get(a).remove(b);
		cons.get(b).remove(a); // also returns r
		
		if( cons.get(a).isEmpty() ) 
			cons.remove(a);
		
		if( cons.get(b).isEmpty() ) 
			cons.remove(b);
					
		//if( r == oldest ) oldest = null;
		return r[0];
	}
	
	// quite slow
	public void purge( int max ) {
		//if( oldest != null && oldest[0] < max ) return;
				
		Set<double[]> toRemove2 = new HashSet<double[]>();
		for( Entry<double[],Map<double[],int[]>> a : cons.entrySet() ) {
			Set<double[]> toRemove = new HashSet<double[]>();
			for( Entry<double[],int[]> e : a.getValue().entrySet() ) 
				if( e.getValue()[0] > max ) 
					toRemove.add(e.getKey());
			a.getValue().keySet().removeAll(toRemove);
			if( a.getValue().isEmpty() )
				toRemove2.add(a.getKey());
		}			
		cons.keySet().removeAll(toRemove2);
	}
	
	public Set<double[]> getNeighbors( double[] d, int depth ) {
		Set<double[]> l = new HashSet<double[]>();
		if( depth == 0 )
			return l;				
		for( double[] nb : cons.get(d).keySet() ) {
			if( !l.contains(nb ) ) {
				l.add( nb );
				l.addAll( getNeighbors( nb, depth-1 ) );
			}
		}
		return l;
	}
	
	public void increase( int i ) {
		for( Map<double[],int[]> v : cons.values() ) 
			for( int[] a : v.values() ) {
				a[0] = a[0]+1;
				//if( oldest == null || oldest[0] < a[0] ) oldest = a;
			}
	}
	
	Set<double[]> getVertices() { // since it is bidirectional and double-connected, return just the keys is enough
		return cons.keySet();	
	}
}
