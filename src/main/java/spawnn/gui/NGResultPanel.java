package spawnn.gui;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
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
import java.util.Map.Entry;
import java.util.Set;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JToggleButton;
import javax.swing.ListCellRenderer;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import net.miginfocom.swing.MigLayout;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.FeatureStore;
import org.geotools.data.FileDataStoreFactorySpi;
import org.geotools.data.shapefile.ShapefileDataStoreFactory;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.feature.type.Name;

import spawnn.dist.Dist;
import spawnn.gui.ClusterDialogGraph.ClusterAlgorithm;
import spawnn.gui.DistanceDialog.DistMode;
import spawnn.gui.DistanceDialog.StatMode;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphClustering;
import spawnn.utils.SpatialDataFrame;
import edu.uci.ics.jung.algorithms.cluster.EdgeBetweennessClusterer;
import edu.uci.ics.jung.algorithms.cluster.VoltageClusterer;
import edu.uci.ics.jung.graph.Graph;

public class NGResultPanel extends ResultPanel<double[]> implements ActionListener, NeuronSelectedListener<double[]> {

	private static Logger log = Logger.getLogger(NGResultPanel.class);
	private static final long serialVersionUID = -4518072006960672609L;

	private JComboBox<String> vertexComboBox;
	private JComboBox edgeComboBox, colorComboBox, layoutComboBox;
	private JButton btnExpGraph, btnExpMap, colorChooser;
	private JToggleButton selectSingle;
	private GraphPanel pnlGraph;
	MapPanel<double[]> mapPanel;

	private Map<double[], Double> neuronValues;
	private Map<double[], Color> selectedColors = new HashMap<double[], Color>();

	private Graph<double[], double[]> g;
	private Dist<double[]> fDist, gDist;

	private List<double[]> pos;
	private Map<double[], Set<double[]>> bmus;
	private FeatureCollection<SimpleFeatureType, SimpleFeature> fc;
	private List<String> names;

	private static final String RANDOM = "Random", DISTANCE = "Distance...", CLUSTER = "Cluster...", CLUSTER_GRAPH = "Cluster (Graph)...";
	private Frame parent;

	private Color selectedColor = Color.RED;

	public NGResultPanel(Frame parent, SpatialDataFrame orig, List<double[]> samples, Map<double[], Set<double[]>> bmus, Graph<double[], double[]> g, Dist<double[]> fDist, Dist<double[]> gDist, int[] ga) {
		super();
		
		this.parent = parent;
		this.g = g;
		this.bmus = bmus;
		this.fDist = fDist;
		this.gDist = gDist;
		this.pos = new ArrayList<double[]>(g.getVertices()); // bmus might not have non-mapping neurons in it
		this.fc = buildClusterFeatures(orig, samples, bmus, pos);
		this.names = orig.names;

		setLayout(new MigLayout(""));

		vertexComboBox = new JComboBox<String>();
		vertexComboBox.addItem(RANDOM);
		vertexComboBox.addItem(DISTANCE);
		vertexComboBox.addItem(CLUSTER);
		vertexComboBox.addItem(CLUSTER_GRAPH);
		
		vertexComboBox.setRenderer(new ComboSeparatorsRenderer<String>((ListCellRenderer<String>)vertexComboBox.getRenderer()){        
		    @Override
			protected boolean addSeparatorAfter(JList list, String value, int index) {
		    	return CLUSTER_GRAPH.equals(value);
			}                                                                            
		});  
		
		for (String s : orig.names)
			vertexComboBox.addItem(s);
		vertexComboBox.addActionListener(this);
		
		edgeComboBox = new JComboBox<String>();
		edgeComboBox.addItem(GraphPanel.NONE);
		edgeComboBox.addItem(GraphPanel.COUNT);
		edgeComboBox.addItem(GraphPanel.DIST);
		log.debug(Arrays.toString(ga));
		if( ga != null && ga.length > 0 )
			edgeComboBox.addItem(GraphPanel.DIST_GEO);
		edgeComboBox.addActionListener(this);

		colorComboBox = new JComboBox();
		colorComboBox.setModel(new DefaultComboBoxModel(ColorBrewerUtil.ColorMode.values()));
		colorComboBox.addActionListener(this);

		layoutComboBox = new JComboBox();
		layoutComboBox.setModel(new DefaultComboBoxModel(GraphPanel.Layout.values()));
		if (ga == null || ga.length != 2) {
			layoutComboBox.removeItem(GraphPanel.Layout.Geo);
		}
		layoutComboBox.setSelectedItem(GraphPanel.Layout.KamadaKawai);

		layoutComboBox.addActionListener(this);

		btnExpGraph = new JButton("Export gas...");
		btnExpGraph.addActionListener(this);

		colorChooser = new JButton("Select color...");
		colorChooser.setBackground(selectedColor);
		colorChooser.addActionListener(this);
		
		selectSingle = new JToggleButton("Select single");
		selectSingle.addActionListener(this);

		btnExpMap = new JButton("Export map...");
		btnExpMap.addActionListener(this);

		pnlGraph = new GraphPanel(g, ga);
		mapPanel = new MapPanel<double[]>(fc, pos);

		actionPerformed(new ActionEvent(vertexComboBox, 0, DISTANCE));
		Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode)colorComboBox.getSelectedItem());

		pnlGraph.setGridColors(colorMap, selectedColors, neuronValues);
		pnlGraph.addNeuronSelectedListener(this);

		add(vertexComboBox, "split 5");
		add(colorComboBox, "");
		add(edgeComboBox,"");
		add(layoutComboBox, "");
		add(btnExpGraph, "");
				
		add(colorChooser, "split 3");		
		add(selectSingle, "");
		add(btnExpMap, "pushx, wrap");
		
		add(pnlGraph, "w 50%, pushy, grow");
		add(mapPanel, "grow");
		
		mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
		mapPanel.addNeuronSelectedListener(this);
		
		colorComboBox.setSelectedItem(ColorBrewerUtil.ColorMode.Blues);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == colorComboBox) {
			Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode)colorComboBox.getSelectedItem());
			pnlGraph.setGridColors(colorMap, selectedColors, neuronValues);
			mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
		} else if (e.getSource() == layoutComboBox) {
			pnlGraph.setGraphLayout((GraphPanel.Layout) layoutComboBox.getSelectedItem());
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
					pnlGraph.saveImage(fn, "PNG");
				} else if( fc.getFileFilter() == FFilter.epsFilter ) {
					/*String s = fn.getAbsolutePath(); //TODO remove this
					s = s.replaceFirst(".eps", "_legend.eps");
					saveLegend(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File(s), "EPS");
					saveLegend(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File("output/cng_35_cluster_legend.eps"), "EPS");
					saveLegend(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File("output/geosom_4_cluster_legend.eps"), "EPS");
					saveLegend2(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File("output/cng_35_cluster_legend2.eps"), true, true);
					saveLegend2(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File("output/geosom_4_cluster_legend2.eps"), true, false);*/
										
					pnlGraph.saveImage(fn, "EPS");
				} else if( fc.getFileFilter() == FFilter.ngXMLFilter ) {
					try {
						NGUtils.saveGas( g.getVertices(), new FileOutputStream(fn));
					} catch (FileNotFoundException e1) {
						e1.printStackTrace();
					}
				}
			}
		} else if (e.getSource() == colorChooser) {
			selectedColor = JColorChooser.showDialog(this, "Select selection color", selectedColor);
			colorChooser.setBackground(selectedColor);
		} else if (e.getSource() == btnExpMap) {
			JFileChooser fChoser = new JFileChooser("output");

			fChoser.setFileFilter(FFilter.pngFilter);
			fChoser.setFileFilter(FFilter.epsFilter);
			fChoser.setFileFilter(FFilter.shpFilter);

			int state = fChoser.showSaveDialog(this);
			if (state == JFileChooser.APPROVE_OPTION) {
				File fn = fChoser.getSelectedFile();
				if (fChoser.getFileFilter() == FFilter.pngFilter) {
					mapPanel.saveImage(fn, "PNG");
				} else if(fChoser.getFileFilter() == FFilter.epsFilter) {
					mapPanel.saveImage(fn, "EPS");
				} else if (fChoser.getFileFilter() == FFilter.shpFilter) {
					try {
						// ugly but works
						FeatureIterator<SimpleFeature> fit = fc.features();
						while (fit.hasNext()) {
							SimpleFeature sf = fit.next();
							double[] gp = pos.get((Integer) (sf.getAttribute("neuron")));
							sf.setAttribute("nValue", neuronValues.get(gp));
							if (selectedColors.containsKey(gp))
								sf.setAttribute("selected", selectedColors.get(gp).getRGB());
						}
						fit.close();

						Map map = Collections.singletonMap("url", fn.toURI().toURL());
						FileDataStoreFactorySpi factory = new ShapefileDataStoreFactory();
						DataStore myData = factory.createNewDataStore(map);
						myData.createSchema(fc.getSchema());
						Name name = myData.getNames().get(0);
						FeatureStore<SimpleFeatureType, SimpleFeature> store = (FeatureStore<SimpleFeatureType, SimpleFeature>) myData.getFeatureSource(name);

						store.addFeatures(fc);
					} catch (Exception ex) {
						ex.printStackTrace();
					}
				}
			}
		} else if (e.getSource() == vertexComboBox) {
			if (vertexComboBox.getSelectedItem() == RANDOM) {
				List<double[]> rndPos = new ArrayList<double[]>(pos);
				Collections.shuffle(rndPos);

				neuronValues = new HashMap<double[], Double>();
				int k = 0;
				for (double[] d : rndPos)
					neuronValues.put(d, (double) k++);

			} else if (vertexComboBox.getSelectedItem() == DISTANCE ) {
				
				DistanceDialog dd = new DistanceDialog(parent, "Distance...", true, gDist != null );
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
						
						int numSubgraphs = Clustering.getSubGraphs(cm).size();
						if( numSubgraphs == 1 ) {
							Map<double[], Set<double[]>> mst = Clustering.getMinimumSpanningTree(cm, fDist);
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
							clusters = Clustering.cutTree( Clustering.getHierarchicalClusterTree(cm, fDist, type), cd.getNumCluster());
						else
							clusters = Clustering.cutTree( Clustering.getHierarchicalClusterTree(ns, fDist, type), cd.getNumCluster());
					}
					
					Map<double[],Set<double[]>> ll = ResultPanel.prototypeClusterToDataCluster(bmus, clusters);
					log.debug("#Cluster: "+ll.size());
					double wcss = ClusterValidation.getWithinClusterSumOfSuqares(ll.values(), fDist);
					double bcss = ClusterValidation.getBetweenClusterSumOfSuqares(ll.values(), fDist);
					log.debug("Within clusters sum of squares: "+wcss);
					log.debug("Between clusters sum of squares: "+bcss);
					//log.debug("Connectivity: "+ClusterValidation.getConnectivity(ll, fDist, 10));
					log.debug("Dunn Index: "+ClusterValidation.getDunnIndex(ll.values(), fDist));
										
					log.debug("quantization error: "+DataUtils.getMeanQuantizationError(ll, fDist));
					if( gDist != null )
						log.debug("spatial quantization error: "+DataUtils.getMeanQuantizationError(ll, gDist));
					log.debug("Davies-Bouldin Index: "+ClusterValidation.getDaviesBouldinIndex(ll, fDist));
					log.debug("Silhouette Coefficient: "+ClusterValidation.getSilhouetteCoefficient(ll, fDist));
					
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
					
					for( Set<double[]> s : clusters ) {
						// get data mapped by all prototypes in s
						Set<double[]> data = new HashSet<double[]>();
						for( double[] proto : s )
							data.addAll( bmus.get(proto) );
							
						// search mean by data
						double[] mean = null;
						for( Entry<double[],Set<double[]>> en : ll.entrySet()  )
							if( en.getValue().containsAll(data) && data.containsAll(en.getValue())) {
								mean = en.getKey();
								break;
							}
													
						// color protos of s
						double v = means.indexOf(mean);
						for( double[] proto : s ) 
							neuronValues.put(proto, v);				
					}
					
					parent.setCursor(Cursor.getDefaultCursor());
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
					log.debug("#Cluster: "+ll.size());
					log.debug("Within clusters sum of squares: "+ClusterValidation.getWithinClusterSumOfSuqares(ll.values(), fDist));
					log.debug("Between clusters sum of squares: "+ClusterValidation.getBetweenClusterSumOfSuqares(ll.values(), fDist));
					//log.debug("Connectivity: "+ClusterValidation.getConnectivity(ll, fDist, 10));
					log.debug("Dunn Index: "+ClusterValidation.getDunnIndex(ll.values(), fDist));
										
					log.debug("quantization error: "+DataUtils.getMeanQuantizationError(ll, fDist));
					if( gDist != null )
					log.debug("spatial quantization error: "+DataUtils.getMeanQuantizationError(ll, gDist));
					log.debug("Davies-Bouldin Index: "+ClusterValidation.getDaviesBouldinIndex(ll, fDist));
					log.debug("Silhouette Coefficient: "+ClusterValidation.getSilhouetteCoefficient(ll, fDist));
					
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
								for( Set<double[]> s : clusters ) 
									if( s.contains(p ) )
										for( double[] d : s )
											neuronValues.put(d, (double)i);
								
								break;
							}
						}							
					}

					parent.setCursor(Cursor.getDefaultCursor());
				}
			} else { // components
				for (double[] v : pos)
					neuronValues.put(v, v[vertexComboBox.getSelectedIndex() - 4]); // RANDOM, DISTANCE, CLUSTER, CLUSTER ( GRAPH)
			}
			Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode)colorComboBox.getSelectedItem());
			pnlGraph.setGridColors(colorMap, selectedColors, neuronValues);
			mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
		} else if( e.getSource() == edgeComboBox ) {
			pnlGraph.setEdgeStyle( (String)edgeComboBox.getSelectedItem());
		} else if (e.getSource() == selectSingle) {
			if (!mapPanel.selectSingle)
				mapPanel.selectSingle = true;
			else
				mapPanel.selectSingle = false;
		}
	}

	@Override
	public void neuronSelectedOccured(NeuronSelectedEvent<double[]> evt) {
		double[] d = evt.getNeuron();

		if (selectedColors.containsKey(d) && selectedColors.get(d) == selectedColor)
			selectedColors.remove(d);
		else
			selectedColors.put(d, selectedColor);

		Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode)colorComboBox.getSelectedItem());
		pnlGraph.setGridColors(colorMap, selectedColors, neuronValues);
		mapPanel.setGridColors(colorMap, selectedColors, neuronValues);
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
}
