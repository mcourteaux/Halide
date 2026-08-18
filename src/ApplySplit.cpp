#include "ApplySplit.h"
#include "IR.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::map;
using std::string;
using std::vector;

vector<ApplySplitResult> apply_split(const Split &split, const string &prefix,
                                     map<string, Expr> &dim_extent_alignment) {
    vector<ApplySplitResult> result;

    Expr outer = Variable::make(Int(32), prefix + split.outer);
    Expr outer_max = Variable::make(Int(32), prefix + split.outer + ".loop_max");
    switch (split.split_type) {
    case Split::SplitVar: {
        string old_var_name = prefix + split.old_var;
        Expr old_var = Variable::make(Int(32), old_var_name);
        Expr old_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
        Expr old_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");
        Expr old_extent = (old_max - old_min) + 1;

        Expr old_original_max = Variable::make(Int(32), prefix + split.old_var + ".max_original");
        Expr old_original_min = Variable::make(Int(32), prefix + split.old_var + ".min_original");

        dim_extent_alignment[split.inner] = split.factor;

        std::string base_name;
        Expr base, inner;
        Expr base_var;
        if (split.inner.empty()) {
            internal_assert(split.outer.empty());
            // no splitting
            base = old_var;
            base_name = old_var_name;
            base_var = old_var;
            inner = 0;
        } else {
            inner = Variable::make(Int(32), prefix + split.inner);
            base = outer * split.factor + old_min;
            base_name = prefix + split.inner + ".base";
            base_var = Variable::make(Int(32), base_name);
        }

        map<string, Expr>::iterator iter = dim_extent_alignment.find(split.old_var);

        Split::TailType tail = split.tail_type;

        if ((iter != dim_extent_alignment.end()) &&
            is_const_zero(simplify(iter->second % split.factor))) {
            // We have proved that the split factor divides the
            // old extent. No need to adjust the base or add an if
            // statement.
            dim_extent_alignment[split.outer] = iter->second / split.factor;
        } else if (is_negative_const(split.factor) || is_const_zero(split.factor)) {
            user_error << "Can't split " << split.old_var << " by " << split.factor
                       << ". Split factors must be strictly positive\n";
        } else if (split.inner.empty()) {
            // Not a split: guard-only
        } else if (tail == Split::TailType::RoundUp) {
            // Nothing to be done here.
        } else if (tail == Split::TailType::ShiftInwards) {
            // Adjust the base downwards to not compute off the
            // end of the realization.

            // We'll only mark the base as likely (triggering a loop
            // partition) if we're at or inside the innermost
            // non-trivial loop.
            base = likely(base);
            base = Min::make(base, old_original_max + (1 - split.factor));
            base = Max::make(base, old_original_min);  // Align_bounds might extend the min down
        } else {
            internal_error << "Unhandled TailType";
        }
#if 0
        } else if (tail == TailStrategy::ShiftInwardsAndBlend) {
            Expr old_base = base;
            base = likely(base);
            base = Min::make(base, old_max + (1 - split.factor));
            // Make a mask which will be a loop invariant if inner gets
            // vectorized, and apply it if we're in the tail.
            Expr unwanted_elems = (-old_extent) % split.factor;
            Expr mask = inner >= unwanted_elems;
            mask = select(base == old_base, likely(const_true()), mask);
            result.emplace_back(mask, ApplySplitResult::BlendProvides);
        } else if (tail == TailStrategy::RoundUpAndBlend) {
            Expr unwanted_elems = (-old_extent) % split.factor;
            Expr mask = inner < split.factor - unwanted_elems;
            mask = select(outer < outer_max, likely(const_true()), mask);
            result.emplace_back(mask, ApplySplitResult::BlendProvides);
        } else {
            internal_assert(tail == TailStrategy::RoundUp);
        }
#endif

        if (split.guard_type == Split::GuardType::NoGuard) {
            // Nothing to do here!
            internal_assert(!split.inner.empty()) << "Cannot define a guard-only split without Guard";
        } else if (split.guard_type == Split::GuardType::GuardWithIf || split.guard_type == Split::GuardType::PredicateLoads || split.guard_type == Split::GuardType::PredicateStores) {
            // Bounds inference has trouble exploiting an if
            // condition. We'll directly tell it that the loop
            // variable is bounded above by the original loop max by
            // replacing the variable with a promise-clamped version
            // of it.
            Expr guarded = promise_clamped(old_var, old_original_min, old_original_max);
            string guarded_var_name = prefix + split.old_var + ".guarded";
            Expr guarded_var = Variable::make(Int(32), guarded_var_name);

            ApplySplitResult::Type predicate_type, substitution_type;
            switch (split.guard_type) {
            case Split::GuardType::GuardWithIf:
                substitution_type = ApplySplitResult::Substitution;
                predicate_type = ApplySplitResult::Predicate;
                break;
            case Split::GuardType::PredicateLoads:
                substitution_type = ApplySplitResult::SubstitutionInCalls;
                predicate_type = ApplySplitResult::PredicateCalls;
                break;
            case Split::GuardType::PredicateStores:
                substitution_type = ApplySplitResult::SubstitutionInProvides;
                predicate_type = ApplySplitResult::PredicateProvides;
                break;

            default:
                internal_assert(false) << "Already handled";
                break;
            }

            // Inject the if condition *after* doing the substitution
            // for the guarded version.
            result.emplace_back(prefix + split.old_var, guarded_var, substitution_type);
            result.emplace_back(guarded_var_name, guarded, ApplySplitResult::LetStmt);
            result.emplace_back(likely(old_original_min <= old_var && old_var <= old_original_max), predicate_type);
        } else if (split.guard_type == Split::GuardType::Blend) {
            result.emplace_back(likely(old_original_min <= old_var && old_var <= old_original_max), ApplySplitResult::BlendProvides);
        }

        if (!split.inner.empty()) {
            // Define the original variable as the base value computed above plus the inner loop variable.
            result.emplace_back(old_var_name, base_var + inner, ApplySplitResult::LetStmt);
            result.emplace_back(base_name, base, ApplySplitResult::LetStmt);
        }
    } break;
    case Split::FuseVars: {
        // Define the inner and outer in terms of the fused var
        Expr fused = Variable::make(Int(32), prefix + split.old_var);
        Expr inner_min = Variable::make(Int(32), prefix + split.inner + ".loop_min");
        Expr inner_max = Variable::make(Int(32), prefix + split.inner + ".loop_max");
        Expr outer_min = Variable::make(Int(32), prefix + split.outer + ".loop_min");

        const Expr &factor = (inner_max - inner_min) + 1;
        Expr inner = fused % factor + inner_min;
        Expr outer = fused / factor + outer_min;

        result.emplace_back(prefix + split.inner, inner, ApplySplitResult::Substitution);
        result.emplace_back(prefix + split.outer, outer, ApplySplitResult::Substitution);
        result.emplace_back(prefix + split.inner, inner, ApplySplitResult::LetStmt);
        result.emplace_back(prefix + split.outer, outer, ApplySplitResult::LetStmt);

        // Maintain the known size of the fused dim if
        // possible. This is important for possible later splits.
        map<string, Expr>::iterator inner_dim = dim_extent_alignment.find(split.inner);
        map<string, Expr>::iterator outer_dim = dim_extent_alignment.find(split.outer);
        if (inner_dim != dim_extent_alignment.end() &&
            outer_dim != dim_extent_alignment.end()) {
            dim_extent_alignment[split.old_var] = inner_dim->second * outer_dim->second;
        }
    } break;
    case Split::RenameVar:
        result.emplace_back(prefix + split.old_var, outer, ApplySplitResult::Substitution);
        result.emplace_back(prefix + split.old_var, outer, ApplySplitResult::LetStmt);
        break;
    }

    return result;
}

vector<std::pair<string, Expr>> compute_loop_bounds_after_split(const Split &split, const string &prefix) {
    // Define the bounds on the split dimensions using the bounds on the function args.
    vector<std::pair<string, Expr>> let_stmts;

    Expr old_var_max = Variable::make(Int(32), prefix + split.old_var + ".loop_max");
    Expr old_var_min = Variable::make(Int(32), prefix + split.old_var + ".loop_min");
    switch (split.split_type) {
    case Split::SplitVar: {
        if (split.inner.empty()) {
            // no loop bounds needed for a guard-only.
            break;
        }
        Expr inner_max = simplify(split.factor - 1);
        Expr outer_max = simplify((old_var_max - old_var_min) / split.factor);
        let_stmts.emplace_back(prefix + split.inner + ".loop_min", 0);
        let_stmts.emplace_back(prefix + split.inner + ".loop_max", inner_max);
        let_stmts.emplace_back(prefix + split.outer + ".loop_min", 0);
        let_stmts.emplace_back(prefix + split.outer + ".loop_max", outer_max);
    } break;
    case Split::FuseVars: {
        // Define bounds on the fused var using the bounds on the inner and outer
        Expr inner_min = Variable::make(Int(32), prefix + split.inner + ".loop_min");
        Expr inner_max = Variable::make(Int(32), prefix + split.inner + ".loop_max");
        Expr outer_min = Variable::make(Int(32), prefix + split.outer + ".loop_min");
        Expr outer_max = Variable::make(Int(32), prefix + split.outer + ".loop_max");
        Expr fused_extent = (inner_max - inner_min + 1) * (outer_max - outer_min + 1);
        let_stmts.emplace_back(prefix + split.old_var + ".loop_min", 0);
        let_stmts.emplace_back(prefix + split.old_var + ".loop_max", simplify(fused_extent - 1));
    } break;
    case Split::RenameVar:
        let_stmts.emplace_back(prefix + split.outer + ".loop_min", old_var_min);
        let_stmts.emplace_back(prefix + split.outer + ".loop_max", old_var_max);
        break;
    }

    return let_stmts;
}

}  // namespace Internal
}  // namespace Halide
