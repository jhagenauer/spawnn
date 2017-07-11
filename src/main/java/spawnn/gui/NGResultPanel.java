package spawnn.gui;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.BorderFactory;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.ListCellRenderer;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import edu.uci.ics.jung.algorithms.cluster.EdgeBetweennessClusterer;
import edu.uci.ics.jung.algorithms.cluster.VoltageClusterer;
import edu.uci.ics.jung.graph.Graph;
import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.gui.ClusterDialogGraph.ClusterAlgorithm;
import spawnn.gui.DistanceDialog.DistMode;
import spawnn.gui.DistanceDialog.StatMode;
import spawnn.gui.NeuronVisPanel.ImageMode;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphClustering;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class NGResultPanel extends ResultPanel<double[]> {

	private static Logger log = Logger.getLogger(NGResultPanel.class);
	private static final long serialVersionUID = -4518072006960672609L;

	private JComboBox<String> vertexComboBox;
	private JComboBox edgeComboBox, gridLayoutComboBox;
	private JButton btnExpGraph;
	private GraphPanel graphPanel;

	private Graph<double[], double[]> g;

	private static final String RANDOM = "Random", DISTANCE = "Distance...", CLUSTER = "Cluster...", CLUSTER_GRAPH = "Cluster (Graph)...";
	
	private Object currentVertexComboBox;
	
	public NGResultPanel(Frame parent, SpatialDataFrame orig, List<double[]> samples, Map<double[], Set<double[]>> bmus, Graph<double[], double[]> g, Dist<double[]> fDist, Dist<double[]> gDist, int[] fa, int[] ga, boolean wmc) {
		super(parent,orig,samples,bmus,new ArrayList<double[]>(g.getVertices()),fDist,gDist);
		String st = "Quantization error: "+DataUtils.getMeanQuantizationError(bmus, fDist);
		if( gDist != null )
			st += ", Spatial quantization error: "+DataUtils.getMeanQuantizationError(bmus, gDist);
		infoField.setText(st);
		this.g = g;
		
		setLayout(new MigLayout(""));

		vertexComboBox = new JComboBox<String>();
		vertexComboBox.addItem(RANDOM);
		vertexComboBox.addItem(DISTANCE);
		vertexComboBox.addItem(CLUSTER);
		vertexComboBox.addItem(CLUSTER_GRAPH);
		
		vertexComboBox.setRenderer(new ComboSeparatorsRendererString((ListCellRenderer<String>)vertexComboBox.getRenderer()){        
		    @Override
			protected boolean addSeparatorAfter(JList list, String value, int index) {
		    	return CLUSTER_GRAPH.equals(value);
			}                                                                            
		});  
		vertexComboBox.setBorder(BorderFactory.createTitledBorder("Neuron"));
		
		Set<Integer> used = new HashSet<Integer>();
		for( int i : fa )
			used.add(i);
		for( int i : ga )
			used.add(i);
		for( int i = 0; i < orig.names.size(); i++ ) {
			String s = orig.names.get(i);
			if( used.contains(i))
				s+="*";
			vertexComboBox.addItem(s);
		}
		if( wmc )
			for( int i = 0; i < orig.names.size(); i++ ) {
				String s = orig.names.get(i);
				if( used.contains(i))
					s+="*";
				s+= " (ctx)";
				vertexComboBox.addItem(s);
			}
				
		vertexComboBox.addActionListener(this);
		currentVertexComboBox = vertexComboBox.getSelectedItem();
		
		edgeComboBox = new JComboBox<String>();
		edgeComboBox.addItem(GraphPanel.NONE);
		edgeComboBox.addItem(GraphPanel.COUNT);
		edgeComboBox.addItem(GraphPanel.DIST);
		log.debug(Arrays.toString(ga));
		if( ga != null && ga.length > 0 )
			edgeComboBox.addItem(GraphPanel.DIST_GEO);
		edgeComboBox.setToolTipText("Set edge style.");
		edgeComboBox.addActionListener(this);
		edgeComboBox.setBorder(BorderFactory.createTitledBorder("Edge"));

		gridLayoutComboBox = new JComboBox();
		gridLayoutComboBox.setModel(new DefaultComboBoxModel(GraphPanel.Layout.values()));
		if (ga == null || ga.length != 2) {
			gridLayoutComboBox.removeItem(GraphPanel.Layout.Geo);
		}
		gridLayoutComboBox.setSelectedItem(GraphPanel.Layout.KamadaKawai);
		gridLayoutComboBox.setToolTipText("Select graph layout.");
		gridLayoutComboBox.addActionListener(this);
		gridLayoutComboBox.setBorder(BorderFactory.createTitledBorder("Graph layout"));

		btnExpGraph = new JButton("Network...");
		btnExpGraph.addActionListener(this);

		graphPanel = new GraphPanel(g, ga, GraphPanel.Layout.KamadaKawai);
		graphPanel.addNeuronSelectedListener(this);

		actionPerformed(new ActionEvent(vertexComboBox, 0, RANDOM));
		//Map<double[], Color> colorMap = updatePanels();

		add(vertexComboBox, "split 5");
		
		JPanel colorPanel = new JPanel(new MigLayout("insets 0, gapy 0"));
		colorPanel.add(colorBrewerBox,"");
		colorPanel.add(colorClassBox,"");
		colorPanel.setBorder(BorderFactory.createTitledBorder("Color scheme"));
		add(colorPanel,"growy");
		
		JPanel selectPanel = new JPanel(new MigLayout("insets 0, gapy 0"));
		selectPanel.add(selectColorButton,"");
		selectPanel.add(selectClearButton,"");
		selectPanel.setBorder(BorderFactory.createTitledBorder("Selection"));
		add(selectPanel,"growy");
		
		add(gridLayoutComboBox, "");
		add(edgeComboBox,"pushx");
						
		JPanel exportPanel = new JPanel(new MigLayout("insets 0, gapy 0"));
		exportPanel.add(btnExpGraph,"");
		exportPanel.add(exportMapButton,"");
		exportPanel.add(exportLegendButton,"");
		exportPanel.setBorder(BorderFactory.createTitledBorder("Export"));
		add(exportPanel,"growy, wrap");
		
		add(graphPanel, "span 2, split 2, w 50%, grow");
		add(mapPanel, "w 50%, grow, wrap");
		add( legendPanel, "span 2, center, wrap");
		//add( infoField, "span 2, growx");
		
		mapPanel.addNeuronSelectedListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == gridLayoutComboBox) {
			graphPanel.setGraphLayout((GraphPanel.Layout) gridLayoutComboBox.getSelectedItem());
		} else if (e.getSource() == btnExpGraph) {
			JFileChooser fc = new JFileChooser("output");

			fc.setFileFilter(FFilter.pngFilter);
			fc.setFileFilter(FFilter.epsFilter);
			fc.setFileFilter(FFilter.ngXMLFilter);
			fc.setFileFilter(FFilter.graphMLFilter);

			int state = fc.showSaveDialog(this);
			if (state == JFileChooser.APPROVE_OPTION) {
				File fn = fc.getSelectedFile();
				if (fc.getFileFilter() == FFilter.graphMLFilter) {
					writeGraphToGraphML(names, g, neuronValues, selectedColors, fn);
				} else if (fc.getFileFilter() == FFilter.pngFilter) {
					graphPanel.saveImage(fn, ImageMode.PNG );
				} else if( fc.getFileFilter() == FFilter.epsFilter ) {
					//String s = fn.getAbsolutePath(); 
					//s = s.replaceFirst(".eps", "_legend.eps");
					//saveLegend( ColorUtils.getColorMap( neuronValues, (ColorBrewer)colorModeBox.getSelectedItem(), false ), neuronValues, new File(s), "EPS" );
										
					graphPanel.saveImage(fn, ImageMode.EPS );
				} else if( fc.getFileFilter() == FFilter.ngXMLFilter ) {
					try {
						NGUtils.saveGas( g.getVertices(), new FileOutputStream(fn));
					} catch (FileNotFoundException e1) {
						e1.printStackTrace();
					}
				}
			}
		} else if (e.getSource() == vertexComboBox) {
			if( !colorClassBox.isEnabled() )
				colorClassBox.setEnabled(true);
			
			if (vertexComboBox.getSelectedItem() == RANDOM) {
				List<double[]> rndPos = new ArrayList<double[]>(pos);
				Collections.shuffle(rndPos);

				neuronValues = new HashMap<double[], Double>();
				int k = 0;
				for (double[] d : rndPos)
					neuronValues.put(d, (double) k++);

			} else if (vertexComboBox.getSelectedItem() == DISTANCE ) {	
				DistanceDialog dd = new DistanceDialog(parent, "Distance...", true, gDist != null );
				if( dd.isOkPressed() ) {
					DistMode dm = dd.getDistMode();
					StatMode sm = dd.getStatMode();
									
					neuronValues = new HashMap<double[], Double>();
					for (double[] v : g.getVertices() ) {
						
						DescriptiveStatistics ds = new DescriptiveStatistics();
						for (double[] nb : g.getNeighbors(v) ) {
							if (dm == DistMode.Normal )
								ds.addValue( fDist.dist(v, nb ) );
							else
								ds.addValue( gDist.dist(v, nb ) );
						}
						if( sm == StatMode.Mean )
							neuronValues.put(v, ds.getMean() );
						else if( sm == StatMode.Median )
							neuronValues.put(v, ds.getPercentile(0.5) );
						else if( sm == StatMode.Variance )
							neuronValues.put(v, ds.getVariance() );
						else if( sm == StatMode.Min )
							neuronValues.put(v, ds.getMin() );
						else if( sm == StatMode.Max )
							neuronValues.put(v, ds.getMax() );
					}
				} else {
					vertexComboBox.setSelectedItem(currentVertexComboBox);
				}
			} else if (vertexComboBox.getSelectedItem() == CLUSTER) {
				ClusterDialogGrid cd = new ClusterDialogGrid(parent, CLUSTER, true, false);

				if (cd.isOkPressed()) {
					parent.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
					
					List<double[]> ns = new ArrayList<double[]>(g.getVertices());
					List<Set<double[]>> clusters = null;

					// connected map
					Map<double[], Set<double[]>> cm = null; 
					if (cd.getConnected()) {
						cm = new HashMap<double[], Set<double[]>>();
						for (double[] v : g.getVertices()) {
							Set<double[]> s = new HashSet<double[]>();
							for (double[] nb : g.getNeighbors(v))
								s.add(nb);
							cm.put(v, s);
						}
					}
					
					if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.kMeans)
						clusters = new ArrayList<Set<double[]>>(Clustering.kMeans(ns, cd.getNumCluster(), fDist).values());
					else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SKATER) {
						
						int numSubgraphs = GraphUtils.getSubGraphs(cm).size();
						if( numSubgraphs == 1 ) {
							Map<double[], Set<double[]>> mst = GraphUtils.getMinimumSpanningTree(cm, fDist);
							clusters = Clustering.skater(mst, cd.getNumCluster() - 1, fDist, 1);
						} else {
							clusters = new ArrayList<Set<double[]>>();
							clusters.add( new HashSet<double[]>(ns) );
							JOptionPane.showMessageDialog(this, "Cannot apply SKATER algorithm. Too many connected components: "+numSubgraphs, "Too many connected components", JOptionPane.ERROR_MESSAGE);
						}
					} else { // hierarchical
						Clustering.HierarchicalClusteringType type = null;
						if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.ALK)
							type = Clustering.HierarchicalClusteringType.average_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.CLK)
							type = Clustering.HierarchicalClusteringType.complete_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SLK)
							type = Clustering.HierarchicalClusteringType.single_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.Ward)
							type = Clustering.HierarchicalClusteringType.ward;

						if (cd.getConnected())
							clusters = Clustering.treeToCluster( Clustering.cutTree( Clustering.getHierarchicalClusterTree(cm, fDist, type), cd.getNumCluster() ) );
						else
							clusters = Clustering.treeToCluster(  Clustering.cutTree( Clustering.getHierarchicalClusterTree(ns, fDist, type), cd.getNumCluster() ) );
					}
					showClusterSummary(parent, ResultPanel.prototypeClusterToDataCluster(bmus, clusters), fDist, gDist);
					
					for( int i = 0; i < clusters.size(); i++ ) 
					for( double[] pt : clusters.get(i) )
						neuronValues.put( pt, (double)i+1);
					
					parent.setCursor(Cursor.getDefaultCursor());
				} else { // ok not pressed
					vertexComboBox.setSelectedItem(currentVertexComboBox);
				}
			} else if (vertexComboBox.getSelectedItem() == CLUSTER_GRAPH) {
				ClusterDialogGraph cd = new ClusterDialogGraph(parent, CLUSTER_GRAPH, true);

 				if (cd.isOkPressed()) {
					parent.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR)); //FIXME does not work. Blame JUNG
					
					List<Set<double[]>> clusters;
					if (cd.getAlgorithm() == ClusterAlgorithm.EdgeBetweeness) {
						EdgeBetweennessClusterer<double[], double[]> clusterer = new EdgeBetweennessClusterer<double[], double[]>(cd.getNumEdgesToRemove());
						clusters = new ArrayList<Set<double[]>>(clusterer.transform(g));

					} else if (cd.getAlgorithm() == ClusterAlgorithm.Voltage) {
						VoltageClusterer<double[], double[]> clusterer = new VoltageClusterer<>(g, cd.getNumCandidates());
						clusters = new ArrayList<Set<double[]>>(clusterer.cluster(cd.getMaxCluster()));
					} else { // multilevel modularity optimization
						
						// build spawnn-graph from jung-graph
						double max = Double.NEGATIVE_INFINITY, maxGeo = Double.NEGATIVE_INFINITY, maxCount = Double.NEGATIVE_INFINITY;
						for (double[] edge : g.getEdges()) {
							maxCount = Math.max(maxCount, edge[0]);
							max = Math.max(max, edge[1]); // dist
							if( edge.length > 2 )
								maxGeo = Math.max(maxGeo, edge[2]); // dist-geo
						}
						
						Map<double[],Map<double[],Double>> graph = new HashMap<double[],Map<double[],Double>>();
						for( double[] v : g.getVertices() ) {
							if( !graph.containsKey(v) )
								graph.put( v, new HashMap<double[],Double>() );
							
							for( double[] nb : g.getNeighbors(v) ) {
								double weight;
								double[] edge = g.findEdge(v, nb);
								if( cd.getWeights() == GraphPanel.COUNT )
									weight = edge[0]/maxCount;
								else if (cd.getWeights() == GraphPanel.DIST)
									weight = 1.0 - edge[1] / max;
								else if (cd.getWeights() == GraphPanel.DIST_GEO && edge.length > 2 )
									weight = 1.0 - edge[2] / maxGeo;
								else
									weight = 1.0;
								
								graph.get(v).put(nb, weight);	
								if( !graph.containsKey(nb) ) // undirected 
									graph.put( nb, new HashMap<double[],Double>() );
								graph.get(nb).put(v, weight);
							}
						}
						
						Map<double[],Integer> map = GraphClustering.multilevelOptimization(graph, cd.getRestarts());
						clusters = new ArrayList<Set<double[]>>( GraphClustering.modulMapToCluster(map) );
												
						/*WeightTransformer<double[]> wt = new WeightTransformer<double[]>();
						double max = Double.NEGATIVE_INFINITY, maxGeo = Double.NEGATIVE_INFINITY, maxCount = Double.NEGATIVE_INFINITY;
						for (double[] edge : g.getEdges()) {
							maxCount = Math.max(maxCount, edge[0]);
							max = Math.max(max, edge[1]); // dist
							maxGeo = Math.max(maxGeo, edge[2]); // dist-geo
						}
						for (double[] edge : g.getEdges())
							if( cd.getWeights() == GraphPanel.COUNT )
								wt.setWeight(edge, edge[0]/maxCount );
							else if (cd.getWeights() == GraphPanel.DIST)
								wt.setWeight(edge, 1.0 - edge[1] / max);
							else if (cd.getWeights() == GraphPanel.DIST_GEO)
								wt.setWeight(edge, 1.0 - edge[2] / maxGeo);
							else
								wt.setWeight(edge, 1.0);
						clusters = new ArrayList<Set<double[]>>(Modularity.transformerToCluster(g, Modularity.multilevelOptimization(g, wt, cd.getRestarts())));*/
					}
					
					Map<double[],Set<double[]>> ll = ResultPanel.prototypeClusterToDataCluster(bmus, clusters);
					showClusterSummary(parent, ll, fDist, gDist);
					
					List<double[]> means = new ArrayList<double[]>(ll.keySet());
					if( gDist != null )
					Collections.sort(means, new Comparator<double[]>() {
						@Override
						public int compare(double[] o1, double[] o2) {
							double[] d = new double[o1.length];
							if (gDist.dist(o1, d) < gDist.dist(o2, d))
								return -1;
							else if (gDist.dist(o1, d) > gDist.dist(o2, d))
								return 1;
							else
								return 0;
						}
					});
					
					neuronValues = new HashMap<double[], Double>();
					for( double[] p : g.getVertices() ) {
						if( neuronValues.containsKey(p) )
							continue;
						
						// find color
						for( int i = 0; i < means.size(); i++ ) {
							if( bmus.containsKey(p) && ll.get(means.get(i)).containsAll(bmus.get(p) ) ) {
								// color all positions
								for( Set<double[]> set : clusters ) 
									if( set.contains(p ) )
										for( double[] d : set )
											neuronValues.put(d, (double)i+1);
								
								break;
							}
						}							
					}

					parent.setCursor(Cursor.getDefaultCursor());
				} else { // clustering canceled
					vertexComboBox.setSelectedItem(currentVertexComboBox);
				}
			} else { // components
				for (double[] v : pos)
					neuronValues.put(v, v[vertexComboBox.getSelectedIndex() - 4]); // RANDOM, DISTANCE, CLUSTER, CLUSTER ( GRAPH)
			}
			
			updatePanels();
			currentVertexComboBox = vertexComboBox.getSelectedItem();
			
		} else if( e.getSource() == edgeComboBox ) {
			graphPanel.setEdgeStyle( (String)edgeComboBox.getSelectedItem());
		} else {
			super.actionPerformed(e);
		}
	}
	
	@Override 
	protected Map<double[], Color> updatePanels() {
		Map<double[],Color> colorMap = super.updatePanels();
		graphPanel.setColors(colorMap, selectedColors, neuronValues);			
		return colorMap;
	}

	public static void writeGraphToGraphML(List<String> names, Graph<double[], double[]> g, Map<double[], Double> neuronValues, Map<double[], Color> selected, File fn) {
		List<double[]> nodes = new ArrayList<double[]>(g.getVertices());
		FileWriter fw = null;

		try {
			fw = new FileWriter(fn);
			XMLOutputFactory factory = XMLOutputFactory.newInstance();
			XMLStreamWriter writer = factory.createXMLStreamWriter(fw);

			writer.writeStartDocument();

			writer.writeStartElement("graphml");
			writer.writeDefaultNamespace("http://graphml.graphdrawing.org/xmlns");

			for (String s : names) {
				writer.writeStartElement("key");
				writer.writeAttribute("id", s);
				writer.writeAttribute("for", "node");
				writer.writeAttribute("attr.name", s);
				writer.writeAttribute("attr.type", "double");
				writer.writeEndElement();
			}

			writer.writeStartElement("key");
			writer.writeAttribute("id", "nValue");
			writer.writeAttribute("for", "node");
			writer.writeAttribute("attr.name", "nValue");
			writer.writeAttribute("attr.type", "double");
			writer.writeEndElement();

			writer.writeStartElement("key");
			writer.writeAttribute("id", "selected");
			writer.writeAttribute("for", "node");
			writer.writeAttribute("attr.name", "selected");
			writer.writeAttribute("attr.type", "int");
			writer.writeEndElement();

			// edge attributes
			int length = g.getEdges().iterator().next().length;
			writer.writeStartElement("key");
			writer.writeAttribute("id", "fDist");
			writer.writeAttribute("for", "edge");
			writer.writeAttribute("attr.name", "fDist");
			writer.writeAttribute("attr.type", "double");
			writer.writeEndElement();
			if (length == 2) {
				writer.writeStartElement("key");
				writer.writeAttribute("id", "gDist");
				writer.writeAttribute("for", "edge");
				writer.writeAttribute("attr.name", "gDist");
				writer.writeAttribute("attr.type", "double");
				writer.writeEndElement();
			}

			writer.writeStartElement("graph");
			writer.writeAttribute("id", "G");
			writer.writeAttribute("edgedefault", "undirected");

			for (double[] n : nodes) {
				writer.writeStartElement("node");
				writer.writeAttribute("id", nodes.indexOf(n) + "");

				for (int i = 0; i < names.size(); i++) {
					writer.writeStartElement("data");
					writer.writeAttribute("key", names.get(i));
					writer.writeCharacters("" + n[i]);
					writer.writeEndElement();
				}

				writer.writeStartElement("data");
				writer.writeAttribute("key", "nValue");
				writer.writeCharacters("" + neuronValues.get(n));
				writer.writeEndElement();

				writer.writeStartElement("data");
				writer.writeAttribute("key", "selected");
				if( selected.containsKey(n) )
					writer.writeCharacters("" + selected.get(n).getRGB());
				else
					writer.writeCharacters("0");
				writer.writeEndElement();

				writer.writeEndElement(); // end node
			}

			int i = 0;
			for (double[] edge : g.getEdges()) {
				writer.writeStartElement("edge");
				writer.writeAttribute("id", i++ + "");
				writer.writeAttribute("source", nodes.indexOf(g.getEndpoints(edge).getFirst()) + "");
				writer.writeAttribute("target", nodes.indexOf(g.getEndpoints(edge).getSecond()) + "");

				// write attributes
				writer.writeStartElement("data");
				writer.writeAttribute("key", "fDist");
				writer.writeCharacters("" + edge[0]);
				writer.writeEndElement(); // end data

				if (length > 1) {
					writer.writeStartElement("data");
					writer.writeAttribute("key", "gDist");
					writer.writeCharacters("" + edge[1]);
					writer.writeEndElement(); // end data
				}

				writer.writeEndElement(); // end edge
			}
			writer.writeEndElement(); // end graph
			writer.writeEndElement(); // end graphml
			writer.writeEndDocument();
			writer.flush();
			writer.close();
			fw.flush();
			fw.close();
		} catch (XMLStreamException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public boolean isClusterVis() {
		return (String)vertexComboBox.getSelectedItem() == CLUSTER || (String)vertexComboBox.getSelectedItem() == CLUSTER_GRAPH;
	}
}
