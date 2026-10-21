"""Search ancestor ownership and durable rooted-graph serialization."""
import copy,gc,pickle,unittest,weakref
import ai

class WeakSearchParents(unittest.TestCase):
    def graph(self):
        root=ai.MCTSNode(None,ai.Side.FREE_PEOPLES)
        chance=ai.MCTSNode(None,root.side,root,ai.Action(-1,'end_turn',{}));chance.is_chance=True
        leaf=ai.MCTSNode(None,root.side,chance)
        root.children=[chance];chance.outcomes=[leaf]
        chance.kernel_sets={'sampled_world':([leaf],1.,0)}
        return root,chance,leaf

    def test_forward_tree_and_active_path_keep_chance_ancestors_alive(self):
        root,chance,leaf=self.graph();path=[root,chance,leaf];reference=weakref.ref(root)
        del root,chance
        self.assertIs(leaf.parent,path[1]);self.assertIs(leaf.parent.parent,path[0])
        self.assertEqual(leaf.parent.action.action_type,'end_turn')
        del path
        self.assertIsNone(reference());self.assertIsNone(leaf.parent)

    def test_detached_cache_keeps_descendants_without_previous_siblings(self):
        root,chance,leaf=self.graph();old=weakref.ref(root)
        chance.parent=None;del root
        self.assertIsNone(old());self.assertIs(leaf.parent,chance)
        self.assertIs(chance.kernel_sets['sampled_world'][0][0],leaf)

    def test_root_and_envelope_deepcopy_pickle_preserve_graph_aliases(self):
        root,chance,leaf=self.graph()
        for clone in (copy.deepcopy,lambda value:pickle.loads(pickle.dumps(value,protocol=5))):
            with self.subTest(clone=clone):
                r,c,l=clone((root,chance,leaf))
                self.assertIsNot(r,root);self.assertIs(c.parent,r);self.assertIs(l.parent,c)
                self.assertIs(r.children[0],c);self.assertIs(c.outcomes[0],l)
                self.assertIs(c.kernel_sets['sampled_world'][0][0],l)
                r=clone(root);self.assertIs(r.children[0].parent,r)
                self.assertIs(r.children[0].outcomes[0].parent,r.children[0])

    def test_legacy_slotted_state_restores_public_parent_without_weakref_pickle(self):
        root,chance,leaf=self.graph()
        old={name:getattr(chance,name) for name in chance.__slots__ if name not in ('_parent_ref','__weakref__')}
        old['parent']=root
        restored=ai.MCTSNode.__new__(ai.MCTSNode);restored.__setstate__((None,old))
        self.assertIs(restored.parent,root);self.assertIs(restored.outcomes[0],leaf)

if __name__=='__main__':unittest.main()
