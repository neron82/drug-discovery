"""PubMed XML parsing regressions using synthetic, local-only citations."""
import pytest

from drugdiscovery.lit.providers import _parse_pubmed_xml


def _pubmed_xml(article_content):
    return f"""<PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123456</PMID>
          <Article>{article_content}</Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">123456</ArticleId>
            <ArticleId IdType="doi">10.0000/synthetic</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>"""


def test_nonempty_abstract_parses_and_preserves_citation_ids():
    paper, = _parse_pubmed_xml(_pubmed_xml("""
      <ArticleTitle>Synthetic citation</ArticleTitle>
      <Abstract><AbstractText>Source abstract text.</AbstractText></Abstract>
    """))

    assert paper.abstract == "Source abstract text."
    assert paper.pmid == "123456"
    assert paper.doi == "10.0000/synthetic"
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/123456/"


def test_mixed_inline_text_preserves_title_and_abstract():
    paper, = _parse_pubmed_xml(_pubmed_xml("""
      <ArticleTitle>Source <i>nested <b>title</b></i> &amp; tail</ArticleTitle>
      <Abstract>
        <AbstractText>Source <i>inline <b>abstract</b></i> with H<sub>2</sub>O &amp; tail.</AbstractText>
      </Abstract>
    """))

    assert paper.title == "Source nested title & tail"
    assert paper.abstract == "Source inline abstract with H2O & tail."


def test_text_entirely_inside_inline_children_is_preserved():
    paper, = _parse_pubmed_xml(_pubmed_xml("""
      <ArticleTitle><i>Entire inline title</i></ArticleTitle>
      <Abstract><AbstractText><b>Entire inline abstract.</b></AbstractText></Abstract>
    """))

    assert paper.title == "Entire inline title"
    assert paper.abstract == "Entire inline abstract."


def test_structured_abstract_preserves_sections_and_only_supplied_labels():
    paper, = _parse_pubmed_xml(_pubmed_xml("""
      <ArticleTitle>Structured citation</ArticleTitle>
      <Abstract>
        <AbstractText Label="Purpose" NlmCategory="BACKGROUND">First <i>section</i>.</AbstractText>
        <AbstractText NlmCategory="METHODS">Unlabelled section.</AbstractText>
        <AbstractText Label="Findings" NlmCategory="RESULTS"><b>Last section.</b></AbstractText>
      </Abstract>
    """))

    assert [line for line in paper.abstract.splitlines() if line] == [
        "Purpose: First section.",
        "Unlabelled section.",
        "Findings: Last section.",
    ]


def test_missing_abstract_remains_valid():
    paper, = _parse_pubmed_xml(_pubmed_xml(
        "<ArticleTitle>Citation without abstract</ArticleTitle>"
    ))

    assert paper.title == "Citation without abstract"
    assert paper.abstract == ""


@pytest.mark.parametrize("title_xml", ["", "<ArticleTitle/>", "<ArticleTitle>  <i> </i>  </ArticleTitle>"])
def test_missing_or_empty_title_is_skipped(title_xml):
    assert _parse_pubmed_xml(_pubmed_xml(f"""
      {title_xml}
      <Abstract><AbstractText>Abstract without a title.</AbstractText></Abstract>
    """)) == []


def test_primary_citation_excludes_alternate_and_reference_text_and_ids():
    paper, = _parse_pubmed_xml("""<PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123456</PMID>
          <Article>
            <ArticleTitle>Primary title</ArticleTitle>
            <Abstract><AbstractText>Primary abstract.</AbstractText></Abstract>
          </Article>
          <OtherAbstract>
            <AbstractText>Alternate abstract.</AbstractText>
          </OtherAbstract>
        </MedlineCitation>
        <PubmedData>
          <ReferenceList><Reference>
            <ArticleIdList><ArticleId IdType="doi">10.0000/reference</ArticleId></ArticleIdList>
            <ArticleTitle>Reference title</ArticleTitle>
            <Abstract><AbstractText>Reference abstract.</AbstractText></Abstract>
          </Reference></ReferenceList>
          <ArticleIdList><ArticleId IdType="doi">10.0000/primary</ArticleId></ArticleIdList>
        </PubmedData>
      </PubmedArticle>
      <PubmedArticle>
        <MedlineCitation><PMID>654321</PMID><Article/></MedlineCitation>
        <PubmedData><ReferenceList><Reference>
          <ArticleTitle>Not a primary title</ArticleTitle>
        </Reference></ReferenceList></PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>""")

    assert paper.title == "Primary title"
    assert paper.abstract == "Primary abstract."
    assert paper.pmid == "123456"
    assert paper.doi == "10.0000/primary"
